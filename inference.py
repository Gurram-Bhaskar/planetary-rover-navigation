"""
inference.py — LLM-Driven Inference Script
Planetary Rover Navigation Simulator · Meta PyTorch Hackathon Round 1
=====================================================================

This script connects to the running OpenEnv Docker container, runs
one episode per task (easy → medium → hard), and uses an LLM via the
OpenAI-compatible API to choose an action at every step.

Environment variables (all required unless marked optional)
-----------------------------------------------------------
  API_BASE_URL   Base URL of the OpenAI-compatible endpoint
                 e.g. "https://api-inference.huggingface.co/v1"
  API_KEY        Bearer token / HF_TOKEN for the LLM endpoint
  MODEL_NAME     Model identifier sent in every chat-completion request
                 e.g. "meta-llama/Llama-3.3-70B-Instruct"
  IMAGE_NAME     Docker image or base URL of the rover environment server
                 e.g. "http://localhost:7860"   (running container)
                 or   "rover-env:latest"         (image name, if using
                 openenv_core.MyEnvV4Env.from_docker_image)

Logging format (mandated by hackathon judges)
---------------------------------------------
  [START] task=<task_id> env=<IMAGE_NAME> model=<MODEL_NAME>
  [STEP]  step=<n> action=<json> reward=<float> done=<bool> error=<str|null>
  [END]   success=<bool> steps=<n> score=<float> rewards=<csv>

Exit codes
----------
  0  all three tasks returned score > 0.0
  1  at least one task scored 0.0  (smoke-test failure)
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
import time
from typing import Any

import aiohttp
from openai import AsyncOpenAI

# =============================================================================
# Environment variable resolution
# =============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "http://localhost:7860")

# Strip trailing slash so we can always append a path safely
_BASE_URL: str = LOCAL_IMAGE_NAME.rstrip("/")

# Task execution order (fixed — easy first builds confidence before hard)
TASKS: list[str] = ["easy", "medium", "hard"]

# LLM generation parameters
LLM_MAX_TOKENS:   int   = 256   # action JSON is short; 256 is generous
LLM_TEMPERATURE:  float = 0.2   # low temperature → more deterministic navigation
LLM_TIMEOUT:      float = 30.0  # seconds before we fall back to a safe action

# Fallback action used when the LLM fails or returns unparseable JSON.
# thrust=0.5 / steering=0.0 / brake=0 is the safest possible motion:
# it moves the rover straight ahead at half speed, burning minimal battery.
FALLBACK_ACTION: dict[str, Any] = {
    "thrust":            0.5,
    "steering":          0.0,
    "brake":             0,
    "vertical_thruster": 0.0,
}

# =============================================================================
# System prompt
# =============================================================================
# Written as a single-shot instruction set.  The model must understand:
#   1. Its identity as a rover navigation controller.
#   2. The exact JSON schema it must output — no prose, no markdown.
#   3. The physical meaning of each action field and its bounds.
#   4. Task-specific strategies baked in so it doesn't need to reason
#      from scratch on every step.
#
# Deliberate choices:
#   - "Respond ONLY with a JSON object" is repeated twice — once in the
#     identity block, once at the end — because LLMs tend to add prose
#     when they feel like explaining themselves.
#   - Bounds are stated as hard limits, not suggestions, to prevent the
#     LLM from generating out-of-range floats that would fail Pydantic
#     validation on the server.
#   - The three task strategies are embedded here so the model has policy
#     knowledge at inference time without needing chain-of-thought.

SYSTEM_PROMPT = """\
You are the autonomous navigation controller for a planetary rover.
Your sole responsibility is to output a single JSON action object every turn.
Respond ONLY with a JSON object — no explanation, no markdown, no extra text.

## Action space (exact JSON schema)
{
  "thrust":            <float in [0.0, 1.0]>,   // forward drive power
  "steering":          <float in [-1.0, 1.0]>,  // -1=hard left, 0=straight, 1=hard right
  "brake":             <int 0 or 1>,             // 1=apply regen braking, 0=drive/coast
  "vertical_thruster": <float in [-0.2, 0.2]>   // vertical adjust (ignored on flat terrain)
}

## Physics you must know
- heading_error = atan2(target_dy, target_dx) - rover_heading
- Normalise heading_error to (-π, π] before using it.
- steering = clamp(heading_error * 2.5, -1.0, 1.0)  → P-controller
- Rover only steers when thrust > 0; steering at thrust=0 has no effect.
- Battery depletes every step; brake=1 halves speed and recovers a tiny
  amount of battery — only useful when you would otherwise overshoot.

## Task strategies

EASY (Flat Plains Transit)
- Compute heading to target, set thrust=1.0, steer to correct heading.
- Battery is abundant; never brake unless target_distance < 3 m.
- Scoring: proximity*0.85 + step_efficiency*0.15 → arrive fast.

MEDIUM (Crater Avoidance)
- A ring of obstacles sits between you and the waypoint.
- If nearest_obstacle_distance < 28 m: steer 90° perpendicular to the
  waypoint direction (pick left or right and hold it) until
  nearest_obstacle_distance > 35 m, then resume beeline.
- Each collision costs -0.06 from the final score.
- Scoring: proximity*0.75 + step_efficiency*0.25 - collision_penalty.

HARD (Battery Sprint)
- Starting battery is only 35%. Drain multiplier is ×4.
- On step 1: compute atan2(target_dy, target_dx), lock that heading, NEVER change it.
- Use thrust=1.0 every step. NEVER brake. NEVER deviate.
- Scoring: proximity*0.65 + battery_efficiency*0.35.

Respond ONLY with the JSON object. Nothing else.\
"""


# =============================================================================
# User prompt builder
# =============================================================================
# Called once per step.  Feeds the LLM the minimum state it needs:
#   - Which task (determines which strategy to apply)
#   - Distance and direction to waypoint (primary navigation signal)
#   - Battery (critical for hard; informational for easy/medium)
#   - Nearest obstacle (determines whether to trigger detour for medium)
#   - Current heading and step budget remaining
#
# We deliberately omit the full obstacle_map array (8×3 floats) from the
# prompt because it adds ~200 tokens and the scalar
# nearest_obstacle_distance is sufficient for the FSM-style detour policy
# we describe in the system prompt.  If you want the full map, add:
#   f"obstacle_map: {obs['obstacle_map']}\n"

def build_user_prompt(
    task_id:  str,
    obs:      dict[str, Any],
    step_num: int,
    max_steps: int,
) -> str:
    """
    Build the per-step user message sent to the LLM.

    Parameters
    ----------
    task_id   : "easy" | "medium" | "hard"
    obs       : the Observation dict returned by /reset or /step
    step_num  : current step index (1-based)
    max_steps : step budget for this task

    Returns
    -------
    A compact plain-text string.  JSON was considered but plain text is
    more token-efficient and models handle it well for numeric inputs.
    """
    # Extract the fields we feed to the model.
    # target_relative gives (dx, dy) — the vector from rover to waypoint.
    # We compute the exact heading error here so the model only needs to
    # clamp and multiply rather than doing trig from scratch.
    dx = obs["target_relative"]["x"]
    dy = obs["target_relative"]["y"]

    # Heading error in radians, normalised to (-π, π]
    target_heading = math.atan2(dy, dx)
    raw_error      = target_heading - obs["rover_heading"]
    # Normalise to (-π, π]
    while raw_error >  math.pi: raw_error -= 2 * math.pi
    while raw_error <= -math.pi: raw_error += 2 * math.pi

    # Pre-compute the P-controller steering value so the model can adopt
    # it directly or nudge it based on obstacle proximity.
    suggested_steering = max(-1.0, min(1.0, raw_error * 2.5))

    terrain_names = {0: "flat/sand", 1: "rocky", 2: "crater_floor", 3: "crater_rim"}
    terrain_label = terrain_names.get(obs["terrain_type"], "unknown")

    return (
        f"TASK: {task_id}\n"
        f"STEP: {step_num}/{max_steps}  "
        f"steps_remaining_norm={obs['steps_remaining_norm']:.3f}\n"
        f"\n"
        f"NAVIGATION\n"
        f"  target_distance      = {obs['target_distance']:.2f} m\n"
        f"  target_dx            = {dx:.2f} m\n"
        f"  target_dy            = {dy:.2f} m\n"
        f"  rover_heading        = {obs['rover_heading']:.4f} rad\n"
        f"  heading_error        = {raw_error:.4f} rad\n"
        f"  suggested_steering   = {suggested_steering:.4f}  "
        f"(P-control, clamp to [-1,1])\n"
        f"\n"
        f"POWER\n"
        f"  battery_level        = {obs['battery_level']:.4f}  "
        f"(0=dead, 1=full)\n"
        f"  battery_drain_rate   = {obs['battery_drain_rate']:.6f} per step\n"
        f"\n"
        f"OBSTACLES\n"
        f"  nearest_obstacle_distance = {obs['nearest_obstacle_distance']:.2f} m  "
        f"(sensor range=50 m; collision at 0.5 m)\n"
        f"  obstacle_count            = {obs['obstacle_count']}\n"
        f"\n"
        f"TERRAIN\n"
        f"  terrain_type   = {obs['terrain_type']} ({terrain_label})\n"
        f"  terrain_slope  = {obs['terrain_slope']}\n"
        f"\n"
        f"Output your action JSON now.\n"
        f"Remember: ONLY a JSON object, no explanation."
    )


# =============================================================================
# LLM action parser
# =============================================================================
# The LLM is instructed to return raw JSON but will sometimes:
#   (a) wrap it in a markdown code block  ```json { ... } ```
#   (b) add a preamble sentence before the JSON
#   (c) return a partial JSON (truncated at max_tokens)
#   (d) use wrong field names (e.g. "steer" instead of "steering")
#   (e) return floats outside the declared bounds
#
# The parser handles all five cases in order, falling back to
# FALLBACK_ACTION only if recovery is impossible.

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def parse_llm_action(raw_text: str) -> tuple[dict[str, Any], str | None]:
    """
    Parse the LLM's raw text response into a valid Action dict.

    Returns
    -------
    (action_dict, error_str)
        action_dict : always a valid action (fallback if parsing failed)
        error_str   : None if parsing succeeded; human-readable error string
                      if we fell back (this goes into [STEP] error=<str>)

    Strategy
    --------
    Step 1 — Strip markdown fences.
        Models trained with RLHF often wrap JSON in ```json ... ```.
        We remove those first.

    Step 2 — Extract the first { ... } block.
        If the model prepended prose ("Sure, here is my action:"), this
        regex finds the JSON object regardless of what came before it.

    Step 3 — Parse JSON.
        Standard json.loads().  If it fails we try a light repair:
        replace single quotes with double quotes (common LLM mistake).

    Step 4 — Field normalisation.
        Accept common aliases (e.g. "steer" → "steering", "gas" → "thrust").
        Any missing required field is filled from FALLBACK_ACTION.

    Step 5 — Bounds clamping.
        Every float/int is clamped to its declared range so the server's
        Pydantic validation never rejects our action.
    """

    # ── Step 1: strip markdown code fences ───────────────────────────────
    # Handles: ```json\n{...}\n``` and ```\n{...}\n```
    stripped = re.sub(r"```(?:json)?\s*", "", raw_text).replace("```", "").strip()

    # ── Step 2: extract first JSON object ────────────────────────────────
    # re.DOTALL because the JSON may span multiple lines.
    match = re.search(r"\{[^{}]*\}", stripped, re.DOTALL)
    if not match:
        # No JSON object found at all — return fallback immediately.
        return FALLBACK_ACTION.copy(), f"no JSON object found in: {raw_text[:80]!r}"

    candidate = match.group(0)

    # ── Step 3: JSON parse with single-quote repair ───────────────────────
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        # Common LLM mistake: single-quoted strings ("'thrust': 0.9")
        repaired = candidate.replace("'", '"')
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError as e:
            return FALLBACK_ACTION.copy(), f"JSON parse failed: {e} | text: {candidate[:80]!r}"

    if not isinstance(parsed, dict):
        return FALLBACK_ACTION.copy(), f"parsed JSON is not a dict: {type(parsed)}"

    # ── Step 4: field normalisation / alias resolution ────────────────────
    # Map common LLM hallucinated field names to canonical ones.
    ALIASES: dict[str, str] = {
        "steer":              "steering",
        "turn":               "steering",
        "yaw":                "steering",
        "gas":                "thrust",
        "throttle":           "thrust",
        "accelerate":         "thrust",
        "speed":              "thrust",
        "brakes":             "brake",
        "braking":            "brake",
        "vert":               "vertical_thruster",
        "vertical":           "vertical_thruster",
        "vertical_thrust":    "vertical_thruster",
        "vthruster":          "vertical_thruster",
    }
    normalised: dict[str, Any] = {}
    for key, val in parsed.items():
        canonical = ALIASES.get(key.lower().strip(), key.lower().strip())
        normalised[canonical] = val

    # Fill any missing required fields from FALLBACK_ACTION.
    # This makes the parser tolerant of partial JSON outputs.
    action: dict[str, Any] = {}
    missing_fields: list[str] = []
    required_fields = ["thrust", "steering", "brake", "vertical_thruster"]

    for field in required_fields:
        if field in normalised:
            action[field] = normalised[field]
        else:
            action[field] = FALLBACK_ACTION[field]
            missing_fields.append(field)

    # ── Step 5: type coercion and bounds clamping ─────────────────────────
    # The LLM may output "1" (string) instead of 1 (int) for brake,
    # or "0.95" (string) for floats.  We coerce first, then clamp.
    coerce_errors: list[str] = []
    try:
        action["thrust"] = _clamp(float(action["thrust"]), 0.0, 1.0)
    except (TypeError, ValueError) as e:
        action["thrust"] = FALLBACK_ACTION["thrust"]
        coerce_errors.append(f"thrust coerce: {e}")

    try:
        action["steering"] = _clamp(float(action["steering"]), -1.0, 1.0)
    except (TypeError, ValueError) as e:
        action["steering"] = FALLBACK_ACTION["steering"]
        coerce_errors.append(f"steering coerce: {e}")

    try:
        # brake must be int 0 or 1.  Accept True/False from JSON booleans.
        raw_brake = action["brake"]
        if isinstance(raw_brake, bool):
            action["brake"] = 1 if raw_brake else 0
        else:
            action["brake"] = int(round(float(raw_brake)))
        action["brake"] = max(0, min(1, action["brake"]))
    except (TypeError, ValueError) as e:
        action["brake"] = FALLBACK_ACTION["brake"]
        coerce_errors.append(f"brake coerce: {e}")

    try:
        action["vertical_thruster"] = _clamp(float(action["vertical_thruster"]), -0.2, 0.2)
    except (TypeError, ValueError) as e:
        action["vertical_thruster"] = FALLBACK_ACTION["vertical_thruster"]
        coerce_errors.append(f"vertical_thruster coerce: {e}")

    # Build error string for [STEP] log — null if everything parsed cleanly.
    error_parts: list[str] = []
    if missing_fields:
        error_parts.append(f"missing_fields={missing_fields}")
    if coerce_errors:
        error_parts.append(f"coerce_errors={coerce_errors}")

    error_str = "; ".join(error_parts) if error_parts else None
    return action, error_str


# =============================================================================
# Logging helpers — exact judge-mandated format
# =============================================================================
# All log lines go to stdout (not stderr) so they are captured by the
# OpenEnv harness.  We flush after every write so lines appear immediately
# even when stdout is line-buffered (e.g. inside Docker).

def log_start(task_id: str) -> None:
    """[START] task=<task> env=<LOCAL_IMAGE_NAME> model=<MODEL_NAME>"""
    print(f"[START] task={task_id} env={LOCAL_IMAGE_NAME} model={MODEL_NAME}", flush=True)


def log_step(
    step_num:   int,
    action:     dict[str, Any],
    reward:     float,
    done:       bool,
    error:      str | None,
) -> None:
    """[STEP] step=<n> action=<json> reward=<float> done=<bool> error=<str|null>"""
    action_json = json.dumps(action, separators=(",", ":"))
    error_val   = f'"{error}"' if error else "null"
    print(
        f"[STEP] step={step_num} "
        f"action={action_json} "
        f"reward={reward:.4f} "
        f"done={str(done).lower()} "
        f"error={error_val}",
        flush=True,
    )


def log_end(
    success:  bool,
    steps:    int,
    score:    float,
    rewards:  list[float],
) -> None:
    """[END] success=<bool> steps=<n> score=<float> rewards=<csv>"""
    rewards_csv = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} "
        f"score={score:.4f} "
        f"rewards={rewards_csv}",
        flush=True,
    )


# =============================================================================
# HTTP helpers — thin async wrappers around aiohttp
# =============================================================================
# We use aiohttp rather than the requests library so the LLM call and
# env poll can be interleaved with asyncio without blocking the event loop.
# All calls are retried once on transient network errors.

async def _http_get(session: aiohttp.ClientSession, path: str, **params) -> dict[str, Any]:
    url = f"{_BASE_URL}{path}"
    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as r:
        r.raise_for_status()
        return await r.json()


async def _http_post(
    session: aiohttp.ClientSession,
    path:    str,
    body:    dict[str, Any],
    **params,
) -> dict[str, Any]:
    url = f"{_BASE_URL}{path}"
    async with session.post(
        url, json=body, params=params,
        timeout=aiohttp.ClientTimeout(total=30),
    ) as r:
        r.raise_for_status()
        return await r.json()


# =============================================================================
# LLM call — one async chat completion per step
# =============================================================================

async def llm_action(
    client:    AsyncOpenAI,
    task_meta: dict[str, Any],
    obs:       dict[str, Any],
    step_num:  int,
) -> tuple[dict[str, Any], str | None]:
    """
    Ask the LLM for one action and parse its response.

    Returns (action_dict, error_str).
    error_str is None on clean parse; a short description on fallback.
    """
    user_msg = build_user_prompt(
        task_id   = task_meta["id"],
        obs       = obs,
        step_num  = step_num,
        max_steps = task_meta["max_steps"],
    )

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model       = MODEL_NAME,
                messages    = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens  = LLM_MAX_TOKENS,
                temperature = LLM_TEMPERATURE,
            ),
            timeout=LLM_TIMEOUT,
        )
        raw_text = response.choices[0].message.content or ""
    except asyncio.TimeoutError:
        return FALLBACK_ACTION.copy(), f"LLM timeout after {LLM_TIMEOUT}s"
    except Exception as e:
        return FALLBACK_ACTION.copy(), f"LLM API error: {type(e).__name__}: {e}"

    return parse_llm_action(raw_text)


# =============================================================================
# Grade a completed episode via /grader
# =============================================================================

async def grade_episode(
    session:    aiohttp.ClientSession,
    episode_id: str,
    task_id:    str,
    last_info:  dict[str, Any],
    last_obs:   dict[str, Any],
) -> float:
    """
    POST /grader with the trajectory summary from the final step's info dict.
    Returns the normalised score [0.0, 1.0].
    Falls back to 0.0 if the grader call fails.
    """
    body = {
        "episode_id":            episode_id,
        "task_id":               task_id,
        "termination_reason":    last_info.get("termination_reason", "unknown"),
        "initial_distance":      last_info.get("initial_distance",   last_obs.get("target_distance", 0.0)),
        "min_distance_achieved": last_info.get("min_distance",       last_obs.get("target_distance", 0.0)),
        "waypoints_reached":     last_info.get("waypoints_hit",      0),
        "total_waypoints":       last_info.get("total_waypoints",    1),
        "steps_taken":           last_info.get("steps",              0),
        "max_steps":             last_info.get("max_steps",          500),
        "battery_remaining":     last_info.get("battery",            last_obs.get("battery_level", 0.0)),
        "collision_count":       last_info.get("collision_count",    0),
    }
    try:
        resp = await _http_post(session, "/grader", body)
        return float(resp.get("score", 0.0))
    except Exception as e:
        print(f"[WARN] /grader call failed: {e}", flush=True)
        return 0.0


# =============================================================================
# Single task episode runner
# =============================================================================

async def run_task(
    session:   aiohttp.ClientSession,
    client:    AsyncOpenAI,
    task_meta: dict[str, Any],
) -> float:
    """
    Run one complete episode for the given task.

    Flow
    ----
    1. POST /reset              → episode_id, initial obs
    2. [START] log
    3. loop until done or truncated:
       a. call LLM for action
       b. POST /step            → obs, reward, done, truncated, info
       c. [STEP] log
    4. POST /grader             → score
    5. [END] log
    6. return score
    """
    task_id   = task_meta["id"]
    max_steps = task_meta["max_steps"]

    # ── 1. Reset ─────────────────────────────────────────────────────────
    reset_resp = await _http_post(session, "/reset", {"task_id": task_id})
    episode_id = reset_resp["episode_id"]
    obs        = reset_resp["obs"]

    # ── 2. START log ──────────────────────────────────────────────────────
    log_start(task_id)

    # Accumulators for [END] log
    rewards:   list[float] = []
    step_num:  int         = 0
    last_info: dict        = {}
    last_obs:  dict        = obs

    # ── 3. Step loop ──────────────────────────────────────────────────────
    while True:
        step_num += 1

        # a. Ask the LLM for an action
        action, parse_error = await llm_action(client, task_meta, obs, step_num)

        # b. Send action to the environment
        try:
            step_resp = await _http_post(
                session, "/step", action,
                episode_id=episode_id,
            )
            obs       = step_resp["obs"]
            reward    = step_resp["reward"]
            done      = step_resp["done"]
            truncated = step_resp["truncated"]
            last_info = step_resp.get("info", {})
            last_obs  = obs
            step_error = parse_error  # propagate LLM parse error if any

        except Exception as e:
            # Env step failed — log the error and terminate this episode.
            reward    = 0.0
            done      = True
            truncated = False
            step_error = f"step HTTP error: {type(e).__name__}: {e}"

        rewards.append(reward)

        # c. [STEP] log
        log_step(step_num, action, reward, done or truncated, step_error)

        if done or truncated:
            break

        # Hard budget guard — should never trigger (server enforces it) but
        # protects against an infinite loop if the server misbehaves.
        if step_num >= max_steps:
            break

    # ── 4. Grade ──────────────────────────────────────────────────────────
    score = await grade_episode(session, episode_id, task_id, last_info, last_obs)

    # ── 5. END log ────────────────────────────────────────────────────────
    success = score > 0.0
    log_end(success=success, steps=step_num, score=score, rewards=rewards)

    return score


# =============================================================================
# Main entry point
# =============================================================================

async def main() -> int:
    """
    Run all three tasks sequentially.
    Returns 0 if every task scored > 0.0; returns 1 otherwise.
    """
    # ── Validate environment variables ────────────────────────────────────
    missing = [v for v in ("API_BASE_URL", "HF_TOKEN", "MODEL_NAME", "LOCAL_IMAGE_NAME")
               if not os.environ.get(v)]
    if missing:
        # Soft warning — we fall back to defaults for most vars, but print
        # the warning so operators know the environment is not fully configured.
        print(
            f"[WARN] The following environment variables are not set and "
            f"defaults will be used: {missing}",
            flush=True,
        )

    if not HF_TOKEN:
        print(
            "[ERROR] API_KEY / HF_TOKEN is required for LLM calls. "
            "Set it as an environment variable and re-run.",
            file=sys.stderr, flush=True,
        )
        return 2

    # ── Initialise clients ────────────────────────────────────────────────
    # AsyncOpenAI is the async variant of the standard openai client.
    # base_url points to any OpenAI-compatible endpoint (HF TGI, vLLM, etc.)
    llm_client = AsyncOpenAI(
        api_key  = HF_TOKEN,
        base_url = API_BASE_URL,
    )

    # aiohttp session shared across all HTTP calls to the env server.
    # Connection pool is reused between tasks to avoid reconnect overhead.
    connector = aiohttp.TCPConnector(limit=4)
    async with aiohttp.ClientSession(connector=connector) as http_session:

        # ── Discover tasks from /tasks ────────────────────────────────────
        try:
            tasks_list = await _http_get(http_session, "/tasks")
        except Exception as e:
            print(f"[ERROR] Could not reach env server at {_BASE_URL}/tasks: {e}",
                  file=sys.stderr, flush=True)
            return 2

        # Index task metadata by id for O(1) lookup
        tasks_by_id: dict[str, dict] = {t["id"]: t for t in tasks_list}

        # ── Run each task ─────────────────────────────────────────────────
        scores: dict[str, float] = {}
        for task_id in TASKS:
            if task_id not in tasks_by_id:
                print(f"[WARN] task_id={task_id!r} not found in /tasks response — skipping.",
                      flush=True)
                scores[task_id] = 0.0
                continue

            task_meta = tasks_by_id[task_id]
            try:
                score = await run_task(http_session, llm_client, task_meta)
            except Exception as e:
                print(f"[ERROR] Unhandled exception in task={task_id}: {e}", flush=True)
                log_end(success=False, steps=0, score=0.0, rewards=[])
                score = 0.0
            scores[task_id] = score

            # Brief pause between tasks to let the server drain any in-flight
            # connections before we start the next episode.
            await asyncio.sleep(0.5)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("INFERENCE COMPLETE", flush=True)
    print("=" * 60, flush=True)
    for task_id, score in scores.items():
        status = "PASS" if score > 0.0 else "FAIL"
        print(f"  [{status}] {task_id:6s}  score={score:.4f}", flush=True)
    print("=" * 60, flush=True)

    all_passed = all(s > 0.0 for s in scores.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
