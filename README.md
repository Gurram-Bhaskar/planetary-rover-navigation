---
title: Planetary Rover Navigation Simulator
emoji: 🪐
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
license: mit
short_description: OpenEnv RL environment — Meta PyTorch Hackathon
---

# Planetary Rover Navigation Simulator

An **OpenEnv-compliant** reinforcement learning environment built for the **Meta PyTorch Hackathon — Round 1**. A rover agent navigates planetary surface terrain, managing battery reserves and avoiding obstacles to reach target waypoints.

The environment is a fully self-contained HTTP microservice that exposes the standard OpenEnv API: `/reset`, `/step`, `/state`, `/tasks`, `/baseline`, and `/grader`.

---

## Quick Start

### Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

### Run the baseline agent (smoke test)

```bash
# Server must be running first
python baseline.py --seed 42

# Or launch the server automatically
python baseline.py --launch --seed 42
```

Exit code `0` = all three tasks scored above `0.0`. Exit code `1` = at least one task failed.

### Run with Docker

```bash
docker build -t rover-env .
docker run -p 7860:7860 rover-env
python baseline.py --url http://localhost:7860 --seed 42
```

### Interactive API docs

Once running, visit `http://localhost:7860/docs` for the full Swagger UI with live endpoint testing.

---

## Environment Overview

| Property | Value |
|---|---|
| World size | 1000 × 1000 m (rover bounded to ±500 m) |
| Timestep | 1 second per `step()` call |
| Max speed | 5.0 m/s at full thrust |
| Waypoint radius | 2.0 m (arrival threshold) |
| Collision radius | 0.5 m (obstacle contact) |
| Sensor range | 50.0 m (obstacle detection) |
| Terrain grid | 20 m × 20 m cells, lazy generation |
| Coordinate system | Cartesian, right-hand, Z = terrain height |

Physics are computed in pure Python using Euler integration — no external simulation library required.

---

## Observation Space

Returned as a JSON object by `/reset`, `/state`, and the `obs` field of `/step`.

| Field | Type | Shape | Bounds | Description |
|---|---|---|---|---|
| `rover_position` | `Box` | `[3]` | `[-500, 500]³` | `[x, y, z]` absolute position in metres |
| `rover_heading` | `Box` | `[1]` | `[-π, π]` | Yaw angle in radians (east = 0) |
| `rover_velocity` | `Box` | `[3]` | `[-5, 5]³` | `[vx, vy, vz]` velocity in m/s |
| `target_position` | `Box` | `[3]` | `[-500, 500]³` | Active waypoint absolute position |
| `target_relative` | `Box` | `[3]` | `[-1000, 1000]³` | Vector from rover to waypoint — use this for goal-conditioned policies |
| `target_distance` | `Box` | `[1]` | `[0, 1414]` | Euclidean distance to active waypoint in metres |
| `waypoints_remaining` | `Discrete` | — | `{0, 1, 2, 3}` | Unvisited waypoints left this episode |
| `obstacle_map` | `Box` | `[8, 3]` | `[-1, 1]` | 8 nearest obstacles as `[dx_norm, dy_norm, dist_norm]`; padded with `dist_norm=1.0` when fewer than 8 in range |
| `obstacle_count` | `Discrete` | — | `{0 … 8}` | Number of obstacles within 50 m sensor range |
| `nearest_obstacle_distance` | `Box` | `[1]` | `[0, 50]` | Raw distance to closest obstacle in metres |
| `battery_level` | `Box` | `[1]` | `[0, 1]` | Normalised remaining battery (0 = dead, 1 = full) |
| `battery_drain_rate` | `Box` | `[1]` | `[0, 1]` | Current drain per step as fraction of total capacity |
| `terrain_type` | `Discrete` | — | `{0, 1, 2, 3}` | Tile under rover: 0=flat, 1=rocky, 2=crater\_floor, 3=crater\_rim |
| `terrain_slope` | `Box` | `[2]` | `[-1, 1]` | `[slope_x, slope_y]` surface normal projections |
| `steps_taken` | `Box` | `[1]` | `[0, 500]` | Steps elapsed this episode |
| `steps_remaining_norm` | `Box` | `[1]` | `[0, 1]` | Remaining step budget normalised to `[0, 1]` |

**Policy tip:** `target_relative` gives you the direct `(dx, dy)` vector every step. Compute `atan2(dy, dx)` to get the heading you need, then steer toward it.

---

## Action Space

Sent as a JSON body to `POST /step?episode_id=<uuid>`.

| Field | Type | Bounds | Description |
|---|---|---|---|
| `thrust` | `Box` float32 | `[0.0, 1.0]` | Forward drive intensity. `0.0` = stopped, `1.0` = full throttle |
| `steering` | `Box` float32 | `[-1.0, 1.0]` | Yaw rate command. `-1.0` = hard left, `0.0` = straight, `1.0` = hard right. Effective yaw rate scales with current thrust |
| `brake` | `Discrete` int32 | `{0, 1}` | Binary regen-braking flag. `1` = halve speed and recover a small amount of battery |
| `vertical_thruster` | `Box` float32 | `[-0.2, 0.2]` | Vertical adjustment for crater terrain. Has no effect and incurs no cost on flat terrain |

**Example action (beeline at full throttle):**

```json
{
  "thrust": 1.0,
  "steering": 0.0,
  "brake": 0,
  "vertical_thruster": 0.0
}
```

---

## Tasks

All three tasks have exactly one waypoint. The rover always spawns at `(0, 0)` heading east.

### Task 1 — Easy: Flat Plains Transit

| Parameter | Value |
|---|---|
| `task_id` | `"easy"` |
| Difficulty | ⭐ |
| Max steps | 200 |
| Starting battery | 100% |
| Drain multiplier | ×1.0 |
| Obstacles | None |
| Terrain | Flat |
| Scoring formula | `proximity × 0.85 + step_efficiency × 0.15` |

Navigate to a single waypoint on flat, open terrain with no obstacles and a full battery. The only challenge is correctly steering toward `target_relative`.

### Task 2 — Medium: Crater Avoidance

| Parameter | Value |
|---|---|
| `task_id` | `"medium"` |
| Difficulty | ⭐⭐ |
| Max steps | 300 |
| Starting battery | 100% |
| Drain multiplier | ×1.0 |
| Obstacles | 1 deterministic crater ring (22 posts, 2 gaps) |
| Terrain | Flat |
| Scoring formula | `proximity × 0.75 + step_efficiency × 0.25 − min(collisions × 0.06, 0.40)` |

A ring of 22 obstacle posts is placed at the midpoint of the rover→waypoint line, blocking the direct path. Two 48° gaps are cut perpendicular to the approach direction. Each collision subtracts 0.06 from the score (capped at −0.40).

**Key observation fields for this task:** `obstacle_map`, `obstacle_count`, `nearest_obstacle_distance`.

### Task 3 — Hard: Battery Sprint

| Parameter | Value |
|---|---|
| `task_id` | `"hard"` |
| Difficulty | ⭐⭐⭐ |
| Max steps | 100 |
| Starting battery | **35%** |
| Drain multiplier | **×4.0** |
| Obstacles | None |
| Terrain | Flat |
| Scoring formula | `proximity × 0.65 + battery_efficiency × 0.35` |

The rover starts with only 35% battery. Combined with a ×4 drain multiplier, a full-throttle beeline exhausts the battery in approximately 8 steps — barely enough to reach the waypoint. Any detour is fatal.

`battery_efficiency = battery_remaining / 0.35` (normalised against starting charge).

---

## API Reference

All endpoints return JSON. The base URL for a running server is `http://localhost:7860`.

### `GET /tasks`

Returns metadata for all three tasks including the full action schema, scoring formula, and policy hints.

```bash
curl http://localhost:7860/tasks
```

### `POST /reset`

Starts a new episode. Returns the initial observation and an `episode_id` required by all subsequent calls.

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42}'
```

**Response fields:** `obs` (full Observation), `episode_id` (UUID string), `task_id`.

### `GET /state`

Returns the current observation without advancing the simulation.

```bash
curl "http://localhost:7860/state?episode_id=<uuid>"
```

### `POST /step`

Applies one action and advances the simulation by one timestep (dt = 1 s).

```bash
curl -X POST "http://localhost:7860/step?episode_id=<uuid>" \
  -H "Content-Type: application/json" \
  -d '{"thrust": 1.0, "steering": 0.0, "brake": 0, "vertical_thruster": 0.0}'
```

**Response fields:** `obs`, `reward` (float), `done` (bool), `truncated` (bool), `info` (dict).

The `info` dict contains grader telemetry ready to pass directly to `/grader`:

```json
{
  "termination_reason": "waypoint_reached | battery_dead | max_steps | unknown",
  "initial_distance": 94.6,
  "min_distance": 0.14,
  "collision_count": 0,
  "waypoints_hit": 1,
  "total_waypoints": 1,
  "steps": 20,
  "max_steps": 200,
  "battery": 0.800
}
```

### `GET /baseline`

Returns the machine-readable environment identity card (name, version, full observation and action space declarations, task list). Used by the OpenEnv registry and auto-validators.

```bash
curl http://localhost:7860/baseline
```

### `POST /grader`

Scores a completed episode. Returns a float in `[0.0, 1.0]`.

All fields can be read directly from the final `step()` `info` dict — no client-side bookkeeping required.

```bash
curl -X POST http://localhost:7860/grader \
  -H "Content-Type: application/json" \
  -d '{
    "episode_id":            "<uuid>",
    "task_id":               "easy",
    "termination_reason":    "waypoint_reached",
    "initial_distance":      94.6,
    "min_distance_achieved": 0.14,
    "waypoints_reached":     1,
    "total_waypoints":       1,
    "steps_taken":           20,
    "max_steps":             200,
    "battery_remaining":     0.800,
    "collision_count":       0
  }'
```

**Response fields:**

| Field | Type | Description |
|---|---|---|
| `score` | float | Final score in `[0.0, 1.0]` |
| `verdict` | string | `WIN`, `WIN_WITH_COLLISIONS`, `PARTIAL_PROGRESS`, `COLLISION_LOSS`, `BATTERY_DEAD`, or `TIMEOUT` |
| `proximity_progress` | float | Raw linear proximity metric. Exactly `0.70` when the rover closed 70% of the gap |
| `score_rationale` | string | One-sentence explanation of the outcome |
| `breakdown` | dict | Per-component scores (keys vary by task) |

---

## Grading

### Scoring formulas

**Easy — Flat Plains Transit**
```
score = proximity × 0.85 + step_efficiency × 0.15
```

**Medium — Crater Avoidance**
```
collision_penalty = min(collision_count × 0.06, 0.40)
score = proximity × 0.75 + step_efficiency × 0.25 − collision_penalty
```

**Hard — Battery Sprint**
```
battery_efficiency = battery_remaining / 0.35
score = proximity × 0.65 + battery_efficiency × 0.35
```

### Shared metrics

**`proximity`** is a strictly linear metric:

```
proximity = 1.0 − (min_distance_achieved / initial_distance)
```

This is exactly `0.70` when the rover closed 70% of the spawn→waypoint gap, `0.0` if it never moved, and overridden to `1.0` on confirmed arrival.

**`step_efficiency`**:
```
step_efficiency = 1.0 − (steps_taken / max_steps)
```

### Score examples

| Scenario | Score |
|---|---|
| Easy: beeline arrival using 50% of budget | 0.85 + 0.075 = **0.925** |
| Easy: arrival using full budget | 0.85 + 0.000 = **0.850** |
| Easy: 70% progress, no arrival | ~0.595–0.700 |
| Medium: arrival, zero collisions | 0.75 + 0.25 = **1.000** |
| Medium: arrival, 3 collisions | 1.00 − 0.18 = **0.820** |
| Medium: stuck in ring, 8+ collisions | ≤ **0.000** |
| Hard: arrival, 50% starting battery left | 0.65 + 0.175 = **0.825** |
| Hard: arrival, battery = 0 on landing | 0.65 + 0.000 = **0.650** |
| Hard: battery dead at 70% progress | 0.455 + 0.000 = **0.455** |

---

## Reward Signal

The step reward returned by `/step` is used for online RL training. It is separate from the grader score.

| Event | Reward | Notes |
|---|---|---|
| Every step | −0.01 | Constant time-pressure penalty |
| Battery drain | −drain × 2.0 | Efficiency incentive |
| **Waypoint reached** | **+100.0** | Massive asymmetric reward to prevent early policy collapse |
| Obstacle collision | −5.0 | Speed zeroed, micro battery penalty |
| Battery depleted | −20.0 | Terminal penalty |
| **Potential-based distance shaping** | `(prev_dist − curr_dist) / initial_dist` | Positive when closing distance; **zero when stationary** (defeats the "stand still" exploit) |
| **Vector-field shaping (near obstacles)** | up to +0.3 | Active within 10 m of obstacles; rewards cosine similarity between rover heading and computed tangent vector (repulsive + attractive gradient blend) |
| Episode complete in < 50% of budget | +5.0 | Efficiency bonus |

---

## File Structure

```
planetary-rover-env/
├── openenv.yaml      # Typed observation + action space declarations
├── main.py           # FastAPI server — physics engine + all routes (1632 lines)
├── inference.py      # LLM-driven inference agent (HF Inference API)
├── train.py          # GRPO training script (Unsloth 4-bit + TRL GRPOTrainer)
├── requirements.txt  # Pinned runtime dependencies
├── Dockerfile        # Two-stage optimised build, port 7860, non-root user
└── README.md         # This file
```

---

## Dependencies

| Package | Version | Role |
|---|---|---|
| `fastapi` | 0.115.6 | ASGI web framework |
| `uvicorn[standard]` | 0.32.1 | ASGI server (uvloop + httptools) |
| `pydantic` | 2.10.3 | Request/response validation |
| `requests` | 2.32.3 | HTTP client in `baseline.py` |

The simulation engine itself uses only Python stdlib (`math`, `random`, `uuid`, `dataclasses`, `enum`).

---

## Baseline Agent Results

Running `python baseline.py --seed 42` against a local server produces the following reference scores:

| Task | Agent strategy | Typical score | Verdict |
|---|---|---|---|
| easy | Beeline: `atan2(dy, dx)` heading lock, `thrust=1.0` | 0.92–0.98 | WIN |
| medium | Two-phase detour: approach → perpendicular → approach | 0.85–0.92 | WIN |
| hard | Heading lock on step 1, never steer again | 0.45–0.65 | WIN / BATTERY_DEAD |

These scores represent deterministic heuristic agents. A trained RL policy should significantly exceed them on all three tasks.

---

## Building Your Own Agent

The minimal loop to run an episode:

```python
import requests, math

BASE = "http://localhost:7860"

# 1. Discover the task
tasks = requests.get(f"{BASE}/tasks").json()

# 2. Reset
resp = requests.post(f"{BASE}/reset", json={"task_id": "easy", "seed": 42}).json()
episode_id = resp["episode_id"]
obs = resp["obs"]

# 3. Step loop
while True:
    dx = obs["target_relative"]["x"]
    dy = obs["target_relative"]["y"]
    heading_error = math.atan2(dy, dx) - obs["rover_heading"]

    action = {
        "thrust":            1.0,
        "steering":          max(-1.0, min(1.0, heading_error * 2.5)),
        "brake":             0,
        "vertical_thruster": 0.0,
    }

    step = requests.post(f"{BASE}/step", json=action,
                         params={"episode_id": episode_id}).json()
    obs = step["obs"]

    if step["done"] or step["truncated"]:
        info = step["info"]
        break

# 4. Grade
grade = requests.post(f"{BASE}/grader", json={
    "episode_id":            episode_id,
    "task_id":               "easy",
    "termination_reason":    info["termination_reason"],
    "initial_distance":      info["initial_distance"],
    "min_distance_achieved": info["min_distance"],
    "waypoints_reached":     info["waypoints_hit"],
    "total_waypoints":       info["total_waypoints"],
    "steps_taken":           info["steps"],
    "max_steps":             info["max_steps"],
    "battery_remaining":     info["battery"],
    "collision_count":       info["collision_count"],
}).json()

print(f"Score: {grade['score']}  Verdict: {grade['verdict']}")
print(f"Rationale: {grade['score_rationale']}")
```

---

## License

MIT
