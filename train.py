"""
train.py — GRPO Training Script for Planetary Rover Navigation
================================================================

Uses Unsloth's FastLanguageModel + TRL's GRPOTrainer to fine-tune
meta-llama/Llama-3.2-1B-Instruct for autonomous rover navigation.

Hardware target : NVIDIA RTX 3050 — strict 6 GB VRAM limit
Quantisation    : 4-bit NF4 via Unsloth
LoRA            : rank 16, attention + MLP projections
GRPO group size : 4 generations per prompt (prevents OOM)

Reward functions
----------------
  1. Format Gatekeeper — validates <action>JSON</action> structure
  2. Environment Reward — POSTs parsed action to local physics server

Prerequisites
-------------
  1. Local server running:
       uvicorn main:app --host 0.0.0.0 --port 7860
  2. Python packages:
       pip install unsloth trl datasets peft accelerate
"""

from __future__ import annotations

import json
import math
import os
import wandb
import re
import sys
import time
import random
import logging
from numbers import Real
from typing import Any

import requests
import torch
from datasets import Dataset

# ---------------------------------------------------------------------------
# Unsloth + TRL imports (deferred to allow --help without GPU)
# ---------------------------------------------------------------------------
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME      = "meta-llama/Llama-3.2-1B-Instruct"
SERVER_URL      = os.getenv("ROVER_SERVER_URL", "http://127.0.0.1:7860")
OUTPUT_DIR      = "./grpo_rover_checkpoints"
SEED            = 42

# Cloud GPU parameters (HF Spaces — migrated from local 6 GB)
MAX_SEQ_LENGTH  = 512          # prompt + completion combined
LORA_RANK       = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.0

# Training hyperparameters
NUM_TRAIN_EPISODES   = 150     # prompts per task × 3 tasks = total dataset
MAX_PROMPT_LENGTH    = 256
MAX_COMPLETION_LENGTH = 256
NUM_GENERATIONS      = int(os.getenv("ROVER_NUM_GENERATIONS", "8"))
LEARNING_RATE        = 1e-6
KL_COEF              = 0.04    # β for KL penalty
NUM_TRAIN_EPOCHS     = 2
PER_DEVICE_BATCH     = 1       # keep at 1 for 6 GB
GRAD_ACCUM_STEPS     = int(os.getenv("ROVER_GRAD_ACCUM_STEPS", "8"))
WARMUP_STEPS         = int(os.getenv("ROVER_WARMUP_STEPS", "10"))
USE_BF16             = os.getenv("ROVER_USE_BF16", "0") == "1"

# Reward tuning
FORMAT_REWARD_GOOD   = 1.0
FORMAT_REWARD_BAD    = 0.0
VERBOSITY_THRESHOLD  = 80      # tokens — a valid <action>{…}</action> is ~30-40
VERBOSITY_PENALTY_K  = 200     # excess tokens before reward → 0

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train")


def _extract_scalar_reward(logs: dict[str, Any]) -> float | None:
    """Return one scalar reward value from a TRL/Trainer log payload."""
    raw_reward = logs.get("reward")
    if isinstance(raw_reward, Real):
        return float(raw_reward)

    reward_terms: list[float] = []
    for key, value in logs.items():
        key_lower = key.lower()
        if "reward" not in key_lower:
            continue
        if any(skip in key_lower for skip in ("std", "min", "max", "var")):
            continue
        if isinstance(value, Real):
            reward_terms.append(float(value))

    if not reward_terms:
        return None

    return sum(reward_terms) / len(reward_terms)


class CompactMetricsCallback(TrainerCallback):
    """Emit a concise log line to simplify screenshot capture in Spaces logs."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control

        loss = logs.get("loss")
        if not isinstance(loss, Real):
            return control

        metrics: dict[str, float] = {"loss": float(loss)}

        reward = _extract_scalar_reward(logs)
        if reward is not None:
            metrics["reward"] = reward

        learning_rate = logs.get("learning_rate")
        if isinstance(learning_rate, Real):
            metrics["lr"] = float(learning_rate)

        compact = {key: round(value, 6) for key, value in metrics.items()}
        log.info("METRICS %s", compact)
        return control


# =============================================================================
# System prompt (compact — must fit within ~90 tokens so user prompt has room)
# =============================================================================

SYSTEM_PROMPT = """\
You are a planetary rover navigation controller.
Respond ONLY with your action inside <action></action> tags as valid JSON.

Action schema:
{"thrust": float[0,1], "steering": float[-1,1], "brake": 0|1, "vertical_thruster": float[-0.2,0.2]}

Key physics:
- heading_error = atan2(target_dy, target_dx) - rover_heading
- steering ≈ clamp(heading_error * 2.5, -1, 1)
- thrust=1.0 for progress; brake=0 unless overshooting
- If nearest_obstacle < 10m, steer perpendicular to dodge\
"""


# =============================================================================
# Compact observation prompt builder
# =============================================================================

def build_compact_prompt(
    task_id:   str,
    obs:       dict[str, Any],
    step_num:  int,
    max_steps: int,
) -> str:
    """
    Build a token-efficient user prompt from an observation dict.
    Designed to fit in ~100–120 tokens so system + user ≤ 256.
    """
    dx = obs["target_relative"]["x"]
    dy = obs["target_relative"]["y"]

    # Pre-compute heading error so the model doesn't need trig
    target_heading = math.atan2(dy, dx)
    raw_error = target_heading - obs["rover_heading"]
    while raw_error >  math.pi: raw_error -= 2 * math.pi
    while raw_error <= -math.pi: raw_error += 2 * math.pi

    suggested_steering = max(-1.0, min(1.0, raw_error * 2.5))

    return (
        f"TASK: {task_id}  STEP: {step_num}/{max_steps}\n"
        f"target_distance={obs['target_distance']:.1f}m "
        f"heading_error={raw_error:.4f}rad\n"
        f"battery={obs['battery_level']:.3f} "
        f"nearest_obstacle={obs['nearest_obstacle_distance']:.1f}m "
        f"terrain={obs['terrain_type']}\n"
        f"suggested_steering={suggested_steering:.4f}\n"
        f"Output your <action> JSON now."
    )


# =============================================================================
# Dataset generation — resets episodes and collects initial observations
# =============================================================================

TASK_MAX_STEPS = {"easy": 200, "medium": 300, "hard": 100}


def _check_server() -> None:
    """Fail fast if the environment server is unreachable."""
    try:
        r = requests.get(f"{SERVER_URL}/tasks", timeout=5)
        r.raise_for_status()
        log.info("Environment server is live at %s", SERVER_URL)
    except Exception as e:
        log.error(
            "Cannot reach environment server at %s — "
            "start it with: uvicorn main:app --host 0.0.0.0 --port 7860",
            SERVER_URL,
        )
        sys.exit(1)


def generate_training_dataset(n_per_task: int = NUM_TRAIN_EPISODES) -> Dataset:
    """
    Generate a training dataset by resetting episodes across all tasks.

    Each row contains:
      prompt   — chat-formatted messages (system + user)
      task_id  — for environment reward replay
      seed     — for environment reward replay
    """
    rows: list[dict[str, Any]] = []

    for task_id in ["easy", "medium", "hard"]:
        max_steps = TASK_MAX_STEPS[task_id]
        for seed in range(n_per_task):
            try:
                resp = requests.post(
                    f"{SERVER_URL}/reset",
                    json={"task_id": task_id, "seed": seed},
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                log.warning("Reset failed (task=%s seed=%d): %s", task_id, seed, e)
                continue

            obs = data["obs"]
            user_msg = build_compact_prompt(task_id, obs, step_num=1, max_steps=max_steps)

            rows.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                "task_id": task_id,
                "seed":    seed,
            })

    random.shuffle(rows)
    log.info("Generated %d training prompts (%d per task × 3 tasks)", len(rows), n_per_task)
    return Dataset.from_list(rows)


# =============================================================================
# Reward Function 1 — Format Gatekeeper
# =============================================================================

# Regex to extract content between <action> and </action> tags
_ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL)

# Required fields and their (min, max) bounds
_ACTION_FIELDS = {
    "thrust":            (0.0,  1.0),
    "steering":          (-1.0, 1.0),
    "brake":             (0,    1),
    "vertical_thruster": (-0.2, 0.2),
}


def _completion_to_text(completion: Any) -> str:
    """Convert TRL completion payloads (str/list/dict) into plain text."""
    if completion is None:
        return ""

    if isinstance(completion, str):
        return completion

    if isinstance(completion, bytes):
        return completion.decode("utf-8", errors="ignore")

    if isinstance(completion, dict):
        for key in ("content", "text", "completion", "generated_text"):
            if key in completion:
                return _completion_to_text(completion[key])
        return str(completion)

    if isinstance(completion, list):
        parts = [_completion_to_text(item) for item in completion]
        return "\n".join(part for part in parts if part)

    return str(completion)


def parse_action_from_completion(completion: Any) -> dict[str, Any] | None:
    """
    Extract and validate an action JSON from <action>…</action> tags.

    Returns the parsed action dict if valid, None otherwise.
    """
    text = _completion_to_text(completion)
    if not text:
        return None

    match = _ACTION_RE.search(text)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    # Validate required fields exist and are numeric
    action: dict[str, Any] = {}
    for field, (lo, hi) in _ACTION_FIELDS.items():
        if field not in parsed:
            return None
        val = parsed[field]
        try:
            if field == "brake":
                val = int(round(float(val)))
            else:
                val = float(val)
        except (TypeError, ValueError):
            return None
        # Reject wildly out-of-range (mild overshoot is clamped, not rejected)
        if val < lo - 0.5 or val > hi + 0.5:
            return None
        # Clamp to valid bounds
        if field == "brake":
            val = max(0, min(1, val))
        else:
            val = max(lo, min(hi, val))
        action[field] = val

    return action


def format_reward_fn(completions: list[Any], **kwargs) -> list[float]:
    """
    Reward Function 1 — The Format Gatekeeper.

    Returns 1.0 if the completion contains valid <action>JSON</action>
    matching the rover action schema.  Returns 0.0 on failure.

    Applies a soft verbosity penalty: completions exceeding
    VERBOSITY_THRESHOLD tokens are penalised linearly, reaching 0
    at VERBOSITY_THRESHOLD + VERBOSITY_PENALTY_K tokens.
    """
    rewards: list[float] = []

    for completion in completions:
        text = _completion_to_text(completion)
        action = parse_action_from_completion(text)
        if action is None:
            rewards.append(FORMAT_REWARD_BAD)
            continue

        # Base reward for valid format
        base = FORMAT_REWARD_GOOD

        # Soft verbosity penalty — count whitespace-split "tokens" as proxy
        # (actual BPE count varies, but this is a stable heuristic)
        token_estimate = len(text.split())
        if token_estimate > VERBOSITY_THRESHOLD:
            excess = token_estimate - VERBOSITY_THRESHOLD
            penalty = max(0.0, 1.0 - excess / VERBOSITY_PENALTY_K)
            base *= penalty

        rewards.append(base)

    return rewards


# =============================================================================
# Reward Function 2 — Environment Reward
# =============================================================================

def environment_reward_fn(completions: list[Any], **kwargs) -> list[float]:
    """
    Reward Function 2 — The Environment.

    For each completion:
      1. Parse the action from <action> tags.
      2. Reset a fresh episode with the same (task_id, seed) as the prompt.
      3. POST the action to /step.
      4. Return the scalar step reward from the physics engine.

    If parsing or HTTP fails, returns 0.0 (neutral — no signal).
    """
    task_ids: list[str] = kwargs.get("task_id", [])
    seeds:    list[int] = kwargs.get("seed", [])

    rewards: list[float] = []

    for i, completion in enumerate(completions):
        # -- Parse action --------------------------------------------------
        action = parse_action_from_completion(completion)
        if action is None:
            rewards.append(0.0)
            continue

        # -- Determine episode parameters ----------------------------------
        # kwargs columns are lists aligned with completions.
        # With num_generations=4, each prompt's metadata is repeated 4 times.
        task_id = task_ids[i] if i < len(task_ids) else "easy"
        seed    = seeds[i]    if i < len(seeds)    else 0

        try:
            # Reset a fresh episode with the same seed → identical starting state
            reset_resp = requests.post(
                f"{SERVER_URL}/reset",
                json={"task_id": task_id, "seed": seed},
                timeout=10,
            )
            reset_resp.raise_for_status()
            episode_id = reset_resp.json()["episode_id"]

            # Step with the generated action
            step_resp = requests.post(
                f"{SERVER_URL}/step",
                json=action,
                params={"episode_id": episode_id},
                timeout=10,
            )
            step_resp.raise_for_status()
            step_data = step_resp.json()

            # Return the scalar reward from the physics engine
            reward = float(step_data.get("reward", 0.0))
            rewards.append(reward)

        except Exception as e:
            log.warning("Environment reward failed (task=%s seed=%d): %s", task_id, seed, e)
            rewards.append(0.0)

    return rewards


# =============================================================================
# Model loading
# =============================================================================

def load_model():
    """
    Load Llama-3.2-1B-Instruct with Unsloth's 4-bit NF4 quantisation
    and attach LoRA adapters to attention + MLP projections.
    """
    log.info("Loading %s with 4-bit NF4 quantisation via Unsloth…", MODEL_NAME)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        # Use a fixed dtype to avoid mixed precision mismatches in LoRA kernels.
        dtype          = torch.bfloat16 if USE_BF16 else torch.float16,
        load_in_4bit   = True,          # NF4 quantisation for 6 GB VRAM
    )

    log.info("Attaching LoRA (rank=%d, alpha=%d) to attention + MLP…", LORA_RANK, LORA_ALPHA)

    model = FastLanguageModel.get_peft_model(
        model,
        r              = LORA_RANK,
        target_modules = [
            # Attention projections
            "q_proj", "k_proj", "v_proj", "o_proj",
            # MLP projections (SwiGLU in Llama)
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
        use_gradient_checkpointing = "unsloth",   # 60% less VRAM
        random_state   = SEED,
    )

    # Ensure pad token is set (required for batched generation)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"    # decoder-only: pad on the left

    vram_gb = torch.cuda.memory_allocated() / 1e9
    log.info("Model loaded. Current VRAM: %.2f GB", vram_gb)

    return model, tokenizer


# =============================================================================
# Training configuration
# =============================================================================

def build_training_config() -> GRPOConfig:
    """Build the GRPOConfig with parameters safe for 6 GB VRAM."""
    # Keep trainer precision aligned with model dtype.
    use_bf16 = USE_BF16

    return GRPOConfig(
        output_dir             = OUTPUT_DIR,

        # ── GRPO-specific ─────────────────────────────────────────────
        num_generations        = NUM_GENERATIONS,     # group size per prompt
        max_prompt_length      = MAX_PROMPT_LENGTH,   # 256 tokens
        max_completion_length  = MAX_COMPLETION_LENGTH,# 256 tokens
        beta                   = KL_COEF,             # KL penalty coeff

        # ── Optimiser ─────────────────────────────────────────────────
        learning_rate          = LEARNING_RATE,        # 1e-6
        lr_scheduler_type      = "cosine",
        warmup_steps           = WARMUP_STEPS,
        max_grad_norm          = 1.0,

        # ── Batch / accumulation ──────────────────────────────────────
        per_device_train_batch_size = PER_DEVICE_BATCH,   # 1 for 6 GB
        # Keep (per_device * grad_accum * world_size) divisible by num_generations.
        gradient_accumulation_steps = GRAD_ACCUM_STEPS,
        num_train_epochs            = NUM_TRAIN_EPOCHS,

        # ── Precision / memory ────────────────────────────────────────
        bf16                   = use_bf16,
        fp16                   = not use_bf16,

        # ── Logging / saving ──────────────────────────────────────────
        logging_steps          = 1,
        save_steps             = 50,
        save_total_limit       = 3,
        report_to              = "wandb",
        run_name               = "openenv-rover-run",
        seed                   = SEED,

        # ── Misc ──────────────────────────────────────────────────────
        remove_unused_columns  = False,                # keep task_id/seed cols
    )


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:
    log.info("=" * 60)
    log.info("GRPO Training — Planetary Rover Navigation")
    log.info("Model : %s", MODEL_NAME)
    log.info("VRAM  : 24 GB+ cloud GPU (4-bit NF4, LoRA r=%d, group=%d)",
             LORA_RANK, NUM_GENERATIONS)
    log.info("Precision: %s", "bf16" if USE_BF16 else "fp16")
    log.info("=" * 60)

    # ── 0. Check server ───────────────────────────────────────────────
    _check_server()

    # ── 1. Load model + tokenizer ─────────────────────────────────────
    model, tokenizer = load_model()

    # ── 2. Generate training dataset ──────────────────────────────────
    log.info("Generating full training dataset from physics engine...")
    train_dataset = generate_training_dataset()

    # ── 3. Build GRPO config ──────────────────────────────────────────
    config = build_training_config()

    # ── 4. Initialise trainer ─────────────────────────────────────────
    log.info("Initialising GRPOTrainer with 2 reward functions…")
    trainer = GRPOTrainer(
        model        = model,
        tokenizer    = tokenizer,
        reward_funcs = [format_reward_fn, environment_reward_fn],
        args         = config,
        train_dataset = train_dataset,
    )
    trainer.add_callback(CompactMetricsCallback())

    # ── 5. Train ──────────────────────────────────────────────────────
    log.info("Starting GRPO training…")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    log.info("Training complete in %.1f minutes.", elapsed / 60)

    # ── 6. Save final adapter ─────────────────────────────────────────
    final_path = os.path.join(OUTPUT_DIR, "final_adapter")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    log.info("Final LoRA adapter saved to %s", final_path)

    # ── 7. VRAM summary ──────────────────────────────────────────────
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    log.info("Peak VRAM usage: %.2f GB", peak_vram)
    if peak_vram > 24.0:
        log.warning("⚠ Peak VRAM exceeded 24 GB! Reduce NUM_GENERATIONS or LORA_RANK.")
    else:
        log.info("✅ VRAM within cloud GPU budget.")


if __name__ == "__main__":
    main()
