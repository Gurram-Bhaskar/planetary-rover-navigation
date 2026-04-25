"""
Planetary Rover Navigation Simulator — OpenEnv Server
======================================================
Phase 3: Task definitions, /tasks endpoint, and starting-battery override.

Architecture
------------
  RoverSim          — pure Python simulation class (no FastAPI dependency)
  SimulationStore   — in-memory dict of live RoverSim instances keyed by episode_id
  FastAPI routes    — thin wrappers that delegate to SimulationStore

Physics overview (2-D plane, Z fixed at terrain height)
--------------------------------------------------------
  Kinematics : Euler integration, dt = 1 s per step
  Steering   : yaw-rate model  ->  heading += steering * MAX_YAW_RATE * thrust
  Velocity   : vx/vy derived from heading + speed each step (no momentum)
  Speed      : thrust maps [0,1] -> [0, MAX_SPEED] m/s; brake halves speed
  Battery    : base drain + terrain multiplier + thrust cost; regen on brake
  Terrain    : seeded height/type grid (20 m cell resolution), lazy evaluation
  Obstacles  : randomly seeded circles; nearest-8 returned per step
  Waypoints  : spawned at episode start; rover must reach within WAYPOINT_RADIUS
  Collision  : nearest obstacle < COLLISION_RADIUS -> penalty, velocity zeroed

All Pydantic models mirror openenv.yaml exactly.
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# =============================================================================
# Physics constants
# =============================================================================

MAX_SPEED        = 5.0    # m/s at full thrust
MAX_YAW_RATE     = 0.6    # rad/s at full thrust + full steering
SENSOR_RANGE     = 50.0   # m  obstacle detection radius
WAYPOINT_RADIUS  = 2.0    # m  "reached" threshold
COLLISION_RADIUS = 0.5    # m  contact threshold
GRID_CELL        = 20.0   # m  terrain tile size
WORLD_HALF       = 500.0  # m  world spans [-500, 500] on each axis
DT               = 1.0    # s  simulation timestep

# Battery drain constants (fraction of capacity per step)
DRAIN_BASE       = 0.002   # idle drain every step
DRAIN_PER_THRUST = 0.008   # additional drain proportional to thrust
DRAIN_TERRAIN    = {0: 0.0, 1: 0.003, 2: 0.005, 3: 0.008}
REGEN_BRAKE      = 0.002   # battery recovered per step when braking

# Task config (mirrors openenv.yaml tasks section)
#
# Three focused, distinct challenges — each has exactly ONE waypoint:
#
#   easy   — flat terrain, no obstacles, normal battery.
#             Baseline navigation; partial credit = pure proximity progress.
#
#   medium — flat terrain, a deterministic crater-rim ring blocks the direct
#             path. Rover must detect and navigate around the obstacle.
#             Collision penalty reduces score; partial credit = proximity
#             minus collision penalty.
#
#   hard   — flat terrain, no obstacles, battery drain ×4.
#             Any significant detour exhausts the battery before arrival.
#             Battery conservation is scored alongside proximity progress.
#
TASK_CONFIG: dict[str, dict] = {
    # ──────────────────────────────────────────────────────────────────────
    # EASY — Baseline navigation, no obstacles, generous battery.
    # The only challenge is pointing toward the waypoint and driving there.
    # Scoring weights proximity heavily (85 %) with a small efficiency bonus.
    # ──────────────────────────────────────────────────────────────────────
    "easy": {
        "display_name":       "Flat Plains Transit",
        "description":        (
            "Navigate to a single waypoint on flat, open terrain. "
            "No obstacles. Full battery. Master the steering model."
        ),
        "difficulty":         1,
        "max_steps":          200,
        "waypoints":          1,
        "terrain_profile":    "flat",
        "obstacle_density":   0.0,
        "battery_drain_mult": 1.0,
        "starting_battery":   1.0,          # full charge — not a constraint
        "world_radius":       120.0,
        "crater_obstacle":    False,
        "scoring_formula":    "proximity*0.85 + step_efficiency*0.15",
        "hints": [
            "Point heading toward target_relative (dx, dy) using atan2.",
            "Maintain thrust=1.0 on a flat beeline for full efficiency score.",
            "battery_drain_rate is low — no need to brake or manage energy.",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    # MEDIUM — A deterministic crater-rim ring bisects the straight-line
    # path to the waypoint.  Two perpendicular gaps allow passage on either
    # side.  Charging straight through triggers collision penalties that
    # subtract directly from the score.
    # Scoring: proximity + efficiency − collision_penalty.
    # ──────────────────────────────────────────────────────────────────────
    "medium": {
        "display_name":       "Crater Avoidance",
        "description":        (
            "A static crater-rim obstacle ring is placed halfway between the "
            "rover and the waypoint, blocking the direct path. "
            "Two side gaps allow passage — detect them via obstacle_map and "
            "steer around the ring. Each collision subtracts 0.06 from your score."
        ),
        "difficulty":         2,
        "max_steps":          300,
        "waypoints":          1,
        "terrain_profile":    "flat",
        "obstacle_density":   0.0,          # no random scatter; crater is placed deterministically
        "battery_drain_mult": 1.0,
        "starting_battery":   1.0,          # full charge — obstacle avoidance is the challenge
        "world_radius":       120.0,
        "crater_obstacle":    True,         # triggers ObstacleField.place_crater_ring() in _make_sim
        "scoring_formula":    "proximity*0.75 + step_efficiency*0.25 - min(collision_count*0.06, 0.40)",
        "hints": [
            "obstacle_map rows are sorted by distance — row 0 is the nearest post.",
            "When nearest_obstacle_distance < 25 m, begin steering perpendicular to target_relative.",
            "The two gaps are perpendicular to the rover→waypoint bearing — aim ±90° off course to find them.",
            "Once past the ring, straighten heading back toward target_relative.",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    # HARD — The rover spawns with only 35 % battery (explicitly overridden
    # below).  Combined with a ×4 drain multiplier, a full-throttle beeline
    # consumes the battery in ≈ 8 steps — just enough to reach a close
    # waypoint if the path is arrow-straight.  Any detour is fatal.
    # Scoring: proximity and battery conservation weighted equally (50/50).
    # ──────────────────────────────────────────────────────────────────────
    "hard": {
        "display_name":       "Battery Sprint",
        "description":        (
            "The rover starts with only 35 % battery charge and drain is "
            "multiplied ×4. Any detour exhausts power before arrival. "
            "Compute the direct vector to the waypoint immediately and commit "
            "to a straight-line full-thrust burn."
        ),
        "difficulty":         3,
        "max_steps":          100,
        "waypoints":          1,
        "terrain_profile":    "flat",
        "obstacle_density":   0.0,
        "battery_drain_mult": 4.0,          # ×4 drain rate
        "starting_battery":   0.35,         # ← OVERRIDE: rover begins at 35 % charge
        "world_radius":       80.0,         # shorter range so a beeline is physically possible
        "crater_obstacle":    False,
        "scoring_formula":    "proximity*0.65 + battery_efficiency*0.35",
        "hints": [
            "Compute target heading = atan2(target_relative.y, target_relative.x) on step 1 and hold it.",
            "Use thrust=1.0 every step — partial throttle wastes proportionally more battery per metre.",
            "Do NOT brake — regen recovers less than the step cost of stopping.",
            "With starting_battery=0.35 and drain_mult=4.0, budget ≈ 8 full-thrust steps.",
        ],
    },
}

# =============================================================================
# Enumerations
# =============================================================================

class TerrainType(IntEnum):
    FLAT_SAND    = 0
    ROCKY        = 1
    CRATER_FLOOR = 2
    CRATER_RIM   = 3


# =============================================================================
# Pydantic models (bounds match openenv.yaml exactly)
# =============================================================================

class Vec3(BaseModel):
    x: float
    y: float
    z: float


class ObstacleEntry(BaseModel):
    dx_norm:   float = Field(..., ge=-1.0, le=1.0)
    dy_norm:   float = Field(..., ge=-1.0, le=1.0)
    dist_norm: float = Field(..., ge=0.0,  le=1.0)


class Observation(BaseModel):
    rover_position:            Vec3
    rover_heading:             float = Field(..., ge=-math.pi, le=math.pi)
    rover_velocity:            Vec3
    target_position:           Vec3
    target_relative:           Vec3
    target_distance:           float = Field(..., ge=0.0, le=1414.0)
    waypoints_remaining:       int   = Field(..., ge=0, le=3)
    obstacle_map:              list[ObstacleEntry] = Field(..., min_length=8, max_length=8)
    obstacle_count:            int   = Field(..., ge=0, le=8)
    nearest_obstacle_distance: float = Field(..., ge=0.0, le=50.0)
    battery_level:             float = Field(..., ge=0.0, le=1.0)
    battery_drain_rate:        float = Field(..., ge=0.0, le=1.0)
    terrain_type:              int   = Field(..., ge=0, le=3)
    terrain_slope:             list[float] = Field(..., min_length=2, max_length=2)
    steps_taken:               float = Field(..., ge=0.0, le=500.0)
    steps_remaining_norm:      float = Field(..., ge=0.0, le=1.0)

    @field_validator("terrain_slope")
    @classmethod
    def slope_bounds(cls, v: list[float]) -> list[float]:
        for val in v:
            if not (-1.0 <= val <= 1.0):
                raise ValueError("terrain_slope components must be in [-1.0, 1.0]")
        return v


class Action(BaseModel):
    thrust:            float = Field(..., ge=0.0,  le=1.0)
    steering:          float = Field(..., ge=-1.0, le=1.0)
    brake:             int   = Field(..., ge=0,    le=1)
    vertical_thruster: float = Field(..., ge=-0.2, le=0.2)


class StepResponse(BaseModel):
    obs:       Observation
    reward:    float
    done:      bool
    truncated: bool
    info:      dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    task_id: str        = Field("easy")
    seed:    int | None = Field(None)


class ResetResponse(BaseModel):
    obs:        Observation
    episode_id: str
    task_id:    str


class TaskMeta(BaseModel):
    id:                 str
    display_name:       str
    description:        str
    difficulty:         int   = Field(..., ge=1, le=3)
    max_steps:          int
    waypoints:          int
    terrain_profile:    str
    obstacle_density:   float
    battery_drain_rate: float
    starting_battery:   float = Field(
        1.0, ge=0.0, le=1.0,
        description="Initial battery at episode start. Hard task overrides to 0.35.",
    )
    target_score:       float
    scoring_formula:    str   = Field(
        "",
        description="Formula showing how /grader computes the score for this task.",
    )
    hints:              list[str] = Field(
        default_factory=list,
        description="Policy hints for building an agent for this task.",
    )
    action_schema:      dict[str, Any] = Field(
        default_factory=dict,
        description="Action space schema: field -> {type, low, high, description}",
    )


class GraderRequest(BaseModel):
    """
    Trajectory summary submitted to /grader at episode end.

    All fields except termination_reason are emitted directly from the
    `info` dict of every StepResponse, so baseline.py can build this
    object from the final step without maintaining separate state.

    Fields
    ------
    episode_id            : UUID returned by /reset
    task_id               : "easy" | "medium" | "hard"
    termination_reason    : why the episode ended — drives the verdict logic
    initial_distance      : straight-line spawn → waypoint distance at reset (m)
                            used as the denominator in proximity_progress
    min_distance_achieved : closest the rover ever got to the active waypoint (m)
                            the numerator in proximity_progress
    waypoints_reached     : count of waypoints the rover entered within 2 m
    total_waypoints       : total waypoints in the task (always 1 for all three tasks)
    steps_taken           : steps elapsed before termination
    max_steps             : step budget for this task
    battery_remaining     : battery level [0.0, 1.0] at episode end
    collision_count       : total obstacle-contact events (medium task penalty)
    """
    episode_id:            str
    task_id:               str

    termination_reason: str = Field(
        "unknown",
        description=(
            "Why the episode ended. One of: "
            "'waypoint_reached' — rover arrived within 2 m of target; "
            "'battery_dead'     — battery hit 0.0 before arrival; "
            "'max_steps'        — step budget exhausted without arrival; "
            "'unknown'          — caller did not specify."
        ),
    )

    initial_distance:      float = Field(..., ge=0.0,
                                         description="Spawn-to-waypoint distance at reset (m)")
    min_distance_achieved: float = Field(..., ge=0.0,
                                         description="Closest approach to waypoint during episode (m)")
    waypoints_reached:     int   = Field(..., ge=0, le=3)
    total_waypoints:       int   = Field(..., ge=1, le=3)
    steps_taken:           int   = Field(..., ge=0, le=500)
    max_steps:             int   = Field(..., ge=1, le=500)
    battery_remaining:     float = Field(..., ge=0.0, le=1.0)
    collision_count:       int   = Field(0, ge=0,
                                         description="Total obstacle-contact events")


class GraderResponse(BaseModel):
    episode_id:        str
    task_id:           str
    score:             float = Field(..., ge=0.0, le=1.0,
                                    description="Final normalised score [0.0, 1.0]")
    verdict:           str   = Field(
        ...,
        description=(
            "Human-readable outcome category. One of: "
            "WIN, WIN_WITH_COLLISIONS, PARTIAL_PROGRESS, "
            "COLLISION_LOSS, BATTERY_DEAD, TIMEOUT."
        ),
    )
    proximity_progress: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Raw linear proximity metric before formula weighting. "
            "Exactly 0.70 when the rover closed 70 % of the spawn→waypoint gap."
        ),
    )
    score_rationale:   str   = Field(
        ...,
        description="One-sentence explanation of how the score was computed.",
    )
    breakdown:         dict[str, float]


class SpaceField(BaseModel):
    name:        str
    type:        str
    shape:       list[int] | None = None
    dtype:       str
    low:         Any | None = None
    high:        Any | None = None
    description: str


class BaselineResponse(BaseModel):
    name:              str
    version:           str
    description:       str
    observation_space: list[SpaceField]
    action_space:      list[SpaceField]
    tasks:             list[str]


# =============================================================================
# Terrain grid  (lazy, seeded, 2-D)
# =============================================================================

@dataclass
class TerrainGrid:
    """
    Seeded 2-D terrain grid. Each cell is GRID_CELL x GRID_CELL metres.
    Cells are generated lazily on first access so we never allocate a full
    1000x1000 grid; only cells the rover visits are populated.

    Cell index: ix = floor((x + WORLD_HALF) / GRID_CELL)
    """
    rng:     random.Random
    profile: str   # "flat" | "rocky" | "crater"

    _types:   dict[tuple[int,int], int]   = field(default_factory=dict)
    _heights: dict[tuple[int,int], float] = field(default_factory=dict)

    _PROFILE_WEIGHTS: dict = field(default_factory=lambda: {
        "flat":   {0: 0.90, 1: 0.08, 2: 0.02, 3: 0.00},
        "rocky":  {0: 0.30, 1: 0.55, 2: 0.10, 3: 0.05},
        "crater": {0: 0.10, 1: 0.20, 2: 0.45, 3: 0.25},
    })

    def _populate(self, ix: int, iy: int) -> None:
        weights = self._PROFILE_WEIGHTS.get(self.profile, self._PROFILE_WEIGHTS["flat"])
        t = self.rng.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]
        base_h = {0: 0.0, 1: self.rng.uniform(0.5, 3.0),
                  2: self.rng.uniform(-4.0, -1.0), 3: self.rng.uniform(2.0, 6.0)}
        self._types[(ix, iy)]   = t
        self._heights[(ix, iy)] = base_h[t]

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return (int((x + WORLD_HALF) / GRID_CELL),
                int((y + WORLD_HALF) / GRID_CELL))

    def terrain_type(self, x: float, y: float) -> int:
        c = self._cell(x, y)
        if c not in self._types:
            self._populate(*c)
        return self._types[c]

    def height(self, x: float, y: float) -> float:
        c = self._cell(x, y)
        if c not in self._heights:
            self._populate(*c)
        return self._heights[c]

    def slope(self, x: float, y: float) -> tuple[float, float]:
        """Finite-difference slope, normalised to [-1, 1]."""
        d = GRID_CELL
        sx = (self.height(x + d, y) - self.height(x - d, y)) / (2 * d)
        sy = (self.height(x, y + d) - self.height(x, y - d)) / (2 * d)
        return max(-1.0, min(1.0, sx)), max(-1.0, min(1.0, sy))


# =============================================================================
# Obstacle field
# =============================================================================

@dataclass
class ObstacleField:
    """
    Circular obstacle field.  Obstacles are point centres; the rover is
    considered to have collided when its position is within COLLISION_RADIUS.

    Two factory constructors:
      generate()           — random scatter (used by medium/hard random modes)
      place_crater_ring()  — deterministic ring bisecting the rover→waypoint
                             straight line (used by medium task)
    """
    obstacles: list[tuple[float, float]] = field(default_factory=list)

    @classmethod
    def generate(cls, rng: random.Random, density: float,
                 world_radius: float, exclusion_radius: float = 15.0) -> "ObstacleField":
        n = int(density * (world_radius ** 2) * 0.05)
        obs: list[tuple[float, float]] = []
        for _ in range(n * 20):
            if len(obs) >= n:
                break
            angle = rng.uniform(0, 2 * math.pi)
            r     = rng.uniform(exclusion_radius, world_radius)
            cx, cy = r * math.cos(angle), r * math.sin(angle)
            if not any(math.hypot(cx - ox, cy - oy) < 3.0 for ox, oy in obs):
                obs.append((cx, cy))
        return cls(obstacles=obs)

    @classmethod
    def place_crater_ring(cls, wx: float, wy: float,
                          ring_radius: float = 18.0,
                          n_posts: int = 22,
                          gap_half_angle: float = 0.42) -> "ObstacleField":
        """
        Build a static crater obstacle for the medium task.

        A ring of `n_posts` obstacle points is placed at `ring_radius` metres
        around the midpoint of the rover→waypoint straight line.  Two symmetric
        gaps (width ~2 * gap_half_angle radians, perpendicular to the approach
        direction) allow the rover to pass on either side — but only if it
        navigates around rather than charging straight through.

        Layout
        ------
        Midpoint  : (wx/2, wy/2)  — centre of the ring
        Gap axis  : perpendicular to the rover→waypoint bearing
                    (so gaps are at ±90° from the direction of travel)
        Gap width : ≈ 48° (gap_half_angle = 0.42 rad ≈ 24° each side)

        With ring_radius = 18 m and n_posts = 22:
          arc spacing ≈ 5.1 m — tight enough that the rover cannot slip between
          posts, but the two gaps are ≈ 15 m wide (passable at max speed).
        """
        mx, my = wx / 2.0, wy / 2.0

        # Bearing from rover (0,0) to waypoint, then rotate 90° for gap axis
        bearing    = math.atan2(wy, wx)
        gap_centre = bearing + math.pi / 2.0   # gaps face left/right of travel

        obs: list[tuple[float, float]] = []
        for i in range(n_posts):
            theta = 2 * math.pi * i / n_posts

            # Angular distance from each gap centre (there are two gaps, ±90°)
            d1 = abs((theta - gap_centre + math.pi) % (2 * math.pi) - math.pi)
            d2 = abs((theta - gap_centre - math.pi + math.pi) % (2 * math.pi) - math.pi)
            in_gap = (d1 < gap_half_angle) or (d2 < gap_half_angle)

            if not in_gap:
                cx = mx + ring_radius * math.cos(theta)
                cy = my + ring_radius * math.sin(theta)
                obs.append((cx, cy))

        return cls(obstacles=obs)

    def nearest_n(self, x: float, y: float, n: int = 8
                  ) -> list[tuple[float, float, float]]:
        """Returns up to n (dx, dy, dist) tuples within SENSOR_RANGE, sorted by dist."""
        within = []
        for cx, cy in self.obstacles:
            dx, dy = cx - x, cy - y
            d = math.hypot(dx, dy)
            if d <= SENSOR_RANGE:
                within.append((dx, dy, d))
        within.sort(key=lambda t: t[2])
        return within[:n]


# =============================================================================
# Core simulation
# =============================================================================

@dataclass
class RoverSim:
    """
    Self-contained 2-D rover simulation.

    State
    -----
    px, py       : position (m)
    heading      : yaw angle (rad), east = 0
    speed        : scalar speed (m/s)
    battery      : [0.0, 1.0]
    steps        : int, incremented each step() call
    done         : True when all waypoints reached or battery dead
    truncated    : True when max_steps reached without done
    waypoints_hit: count of waypoints successfully reached
    """
    task_id:       str
    max_steps:     int
    drain_mult:    float
    terrain:       TerrainGrid
    obstacles:     ObstacleField
    waypoint_list: list[tuple[float, float]]

    # Rover state (all mutable)
    px:           float = 0.0
    py:           float = 0.0
    heading:      float = 0.0
    speed:        float = 0.0
    battery:      float = 1.0
    steps:        int   = 0
    done:         bool  = False
    truncated:    bool  = False
    waypoints_hit: int  = 0
    total_reward: float = 0.0

    # Grading telemetry — populated by _make_sim, updated each step
    initial_distance: float = 0.0   # spawn → first waypoint (set once at reset)
    min_distance:     float = 0.0   # running minimum; drives partial-progress score
    collision_count:  int   = 0     # cumulative obstacle contacts

    # Reward-shaping state — tracks distance at previous step for potential-based shaping
    _prev_distance:   float = 0.0   # set equal to initial_distance at reset

    # ── Sim-to-Real: Domain Randomization (Feature 1) ──────────────────
    # Random per-episode modifier applied to speed in _apply_kinematics.
    # Simulates varying terrain friction / gravity each episode.
    physics_modifier: float = 1.0

    # ── Sim-to-Real: Servo Rate Limiting (Feature 2) ───────────────────
    # Tracks last step's steering so we can clamp Δsteering ≤ 0.5/step.
    previous_steering: float = 0.0

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    @property
    def active_waypoint(self) -> tuple[float, float] | None:
        if self.waypoints_hit < len(self.waypoint_list):
            return self.waypoint_list[self.waypoints_hit]
        return None

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    @staticmethod
    def _wrap(h: float) -> float:
        while h >  math.pi: h -= 2 * math.pi
        while h <= -math.pi: h += 2 * math.pi
        return h

    # -------------------------------------------------------------------
    # Kinematics  (called by step)
    # -------------------------------------------------------------------

    def _apply_kinematics(self, action: Action) -> None:
        """
        1. Heading:  yaw rate scales with thrust so steering only works
                     when the rover is moving (realistic differential drive).
        2. Speed:    set directly from thrust; braking halves it.
                     Terrain slope projects onto heading to add drag/assist.
        3. Position: Euler integration; world-boundary clamped.
        """
        # Heading update
        yaw_rate = action.steering * MAX_YAW_RATE * (action.thrust + 0.1)
        self.heading = self._wrap(self.heading + yaw_rate * DT)

        # Speed update
        if action.brake:
            target_speed = self.speed * 0.5
        else:
            target_speed = action.thrust * MAX_SPEED

        # Slope drag: project (slope_x, slope_y) onto heading direction
        sx, sy = self.terrain.slope(self.px, self.py)
        slope_proj = sx * math.cos(self.heading) + sy * math.sin(self.heading)
        drag = 1.0 - self._clamp(slope_proj * 0.3, -0.3, 0.3)
        # ── Sim-to-Real: Domain Randomization (Feature 1) ──────────────
        # Apply per-episode physics_modifier to simulate terrain friction
        # variation.  Ground-truth physics uses the modified speed.
        self.speed = self._clamp(target_speed * drag * self.physics_modifier, 0.0, MAX_SPEED)

        # Position update with world clamping
        self.px = self._clamp(self.px + self.speed * math.cos(self.heading) * DT,
                              -WORLD_HALF, WORLD_HALF)
        self.py = self._clamp(self.py + self.speed * math.sin(self.heading) * DT,
                              -WORLD_HALF, WORLD_HALF)

    # -------------------------------------------------------------------
    # Battery  (called by step)
    # -------------------------------------------------------------------

    def _update_battery(self, action: Action) -> float:
        """Compute and apply battery drain. Returns drain amount (>= 0)."""
        t_type        = self.terrain.terrain_type(self.px, self.py)
        terrain_drain = DRAIN_TERRAIN.get(t_type, 0.0)
        thrust_drain  = DRAIN_PER_THRUST * action.thrust
        regen         = REGEN_BRAKE if action.brake else 0.0
        drain = max(0.0, (DRAIN_BASE + terrain_drain + thrust_drain - regen) * self.drain_mult)
        self.battery  = self._clamp(self.battery - drain, 0.0, 1.0)
        return drain

    # -------------------------------------------------------------------
    # Collision  (called by step)
    # -------------------------------------------------------------------

    def _check_collision(self) -> tuple[bool, float]:
        """
        Scan all obstacles. Returns (collided, nearest_distance_m).
        On collision: speed zeroed, micro battery penalty applied.
        """
        nearest  = SENSOR_RANGE
        collided = False
        for cx, cy in self.obstacles.obstacles:
            d = math.hypot(cx - self.px, cy - self.py)
            if d < nearest:
                nearest = d
            if d < COLLISION_RADIUS:
                collided = True
        if collided:
            self.speed        = 0.0
            self.battery      = self._clamp(self.battery - 0.01, 0.0, 1.0)
            self.collision_count += 1
        return collided, min(nearest, SENSOR_RANGE)

    # -------------------------------------------------------------------
    # Waypoint check  (called by step)
    # -------------------------------------------------------------------

    def _check_waypoints(self) -> bool:
        """Advance waypoint counter if rover is within WAYPOINT_RADIUS. Returns True on hit."""
        wp = self.active_waypoint
        if wp and math.hypot(wp[0] - self.px, wp[1] - self.py) <= WAYPOINT_RADIUS:
            self.waypoints_hit += 1
            return True
        return False

    # -------------------------------------------------------------------
    # Reward  (called by step)
    # -------------------------------------------------------------------

    def _compute_reward(
        self,
        waypoint_hit: bool,
        collided: bool,
        drain: float,
        prev_dist: float,
    ) -> float:
        """
        Upgraded reward with two anti-exploit shaping mechanisms:

        1. **Potential-Based Reward Shaping (flat plains)**
           Φ(s) = −distance_to_goal.  Shaping = γΦ(s') − Φ(s) ≈ prev_dist − curr_dist.
           If the rover stands still, curr_dist == prev_dist → shaping = 0,
           so the step penalty + battery drain yield a guaranteed net negative.

        2. **Vector-Field Reward Shaping (craters / obstacles)**
           When any obstacle is within 10 m, compute:
             • Attractive gradient  g_a = normalise(goal − pos)
             • Repulsive gradient   g_r = Σ (1/d² − 1/D²) · normalise(pos − obs)
           Blend into a combined desired vector, take its orthogonal tangent
           (so the rover flows *around* obstacles rather than into them),
           and reward based on cosine similarity between the rover's actual
           heading vector and the tangent vector.

        The massive +100.0 asymmetric waypoint reward is preserved to
        anchor the policy toward goal completion.
        """
        r = 0.0

        # ── 0. Constant step cost (time pressure) ──────────────────────
        r -= 0.01

        # ── 1. Battery efficiency penalty ──────────────────────────────
        r -= drain * 2.0
        if self.battery <= 0.0:
            r -= 20.0

        # ── 2. Collision penalty ───────────────────────────────────────
        if collided:
            r -= 5.0

        # ── 3. Waypoint reached — massive asymmetric reward ───────────
        if waypoint_hit:
            r += 100.0

        # ── 4. Potential-based distance shaping ────────────────────────
        #   Φ(s) = −dist  →  F_shape = Φ(s') − Φ(s) = prev_dist − curr_dist
        #   Stationary rover: curr == prev → shaping = 0 → net reward < 0
        wp = self.active_waypoint
        if wp:
            curr_dist = math.hypot(wp[0] - self.px, wp[1] - self.py)
            # Scale by 1/initial_distance so shaping magnitude is
            # independent of spawn distance (reward ∈ roughly [-1, +1])
            scale = 1.0 / max(self.initial_distance, 1.0)
            distance_shaping = (prev_dist - curr_dist) * scale
            r += distance_shaping
        else:
            curr_dist = 0.0

        # ── 5. Vector-field shaping near obstacles (within 10 m) ───────
        INFLUENCE_RADIUS = 10.0
        nearest_obs = self.obstacles.nearest_n(self.px, self.py, 8)
        close_obstacles = [(dx, dy, d) for dx, dy, d in nearest_obs
                           if d < INFLUENCE_RADIUS and d > 1e-6]

        if close_obstacles and wp:
            # 5a. Attractive gradient: unit vector toward goal
            g_ax = wp[0] - self.px
            g_ay = wp[1] - self.py
            g_a_mag = math.hypot(g_ax, g_ay)
            if g_a_mag > 1e-6:
                g_ax /= g_a_mag
                g_ay /= g_a_mag
            else:
                g_ax, g_ay = 0.0, 0.0

            # 5b. Repulsive gradient: sum of inverse-square repulsions
            #     g_r = Σ_i  (1/d_i² − 1/D²) · normalise(pos − obs_i)
            D = INFLUENCE_RADIUS
            g_rx, g_ry = 0.0, 0.0
            for dx, dy, d in close_obstacles:
                # dx, dy point FROM rover TO obstacle; we want FROM obstacle
                repel_x, repel_y = -dx, -dy
                rep_mag = math.hypot(repel_x, repel_y)
                if rep_mag > 1e-6:
                    repel_x /= rep_mag
                    repel_y /= rep_mag
                strength = (1.0 / (d * d)) - (1.0 / (D * D))
                g_rx += strength * repel_x
                g_ry += strength * repel_y

            # 5c. Blend attractive + repulsive into desired vector
            alpha = 0.5   # blending weight for repulsive component
            blend_x = g_ax + alpha * g_rx
            blend_y = g_ay + alpha * g_ry

            # 5d. Compute tangent (90° CCW rotation of the blended vector)
            #     so the rover is guided to flow *around* the obstacle field
            tangent_x = -blend_y
            tangent_y =  blend_x
            t_mag = math.hypot(tangent_x, tangent_y)
            if t_mag > 1e-6:
                tangent_x /= t_mag
                tangent_y /= t_mag

                # 5e. Rover's actual heading unit vector
                hx = math.cos(self.heading)
                hy = math.sin(self.heading)

                # 5f. Cosine similarity (absolute value — either tangent
                #     direction is acceptable, clockwise or counter-clockwise)
                cos_sim = abs(hx * tangent_x + hy * tangent_y)

                # Scale reward by proximity urgency: closer → stronger signal
                min_d = close_obstacles[0][2]   # already sorted ascending
                proximity_weight = 1.0 - (min_d / INFLUENCE_RADIUS)
                r += 0.3 * cos_sim * proximity_weight

        # ── 6. Efficiency bonus: episode done in < 50% of step budget ─
        if (self.waypoints_hit == len(self.waypoint_list)
                and self.steps < self.max_steps * 0.5):
            r += 5.0

        return r

    # -------------------------------------------------------------------
    # Observation builder
    # -------------------------------------------------------------------

    def _build_obs(self) -> Observation:
        # Use last waypoint as reference after all are collected
        wp = self.active_waypoint or self.waypoint_list[-1]
        wx, wy = wp
        dx, dy = wx - self.px, wy - self.py
        dist   = math.hypot(dx, dy)

        # Obstacle sensor: nearest 8
        nearest_raw  = self.obstacles.nearest_n(self.px, self.py, 8)
        nearest_dist = nearest_raw[0][2] if nearest_raw else SENSOR_RANGE

        obs_map: list[ObstacleEntry] = [
            ObstacleEntry(
                dx_norm  = self._clamp(e[0] / SENSOR_RANGE, -1.0, 1.0),
                dy_norm  = self._clamp(e[1] / SENSOR_RANGE, -1.0, 1.0),
                dist_norm= self._clamp(e[2] / SENSOR_RANGE,  0.0, 1.0),
            )
            for e in nearest_raw
        ]
        while len(obs_map) < 8:
            obs_map.append(ObstacleEntry(dx_norm=0.0, dy_norm=0.0, dist_norm=1.0))

        t_type      = self.terrain.terrain_type(self.px, self.py)
        t_height    = self.terrain.height(self.px, self.py)
        slope_x, slope_y = self.terrain.slope(self.px, self.py)

        current_drain = (DRAIN_BASE + DRAIN_TERRAIN.get(t_type, 0.0)) * self.drain_mult
        waypts_rem    = len(self.waypoint_list) - self.waypoints_hit

        # ── Sim-to-Real: Sensor Noise (Feature 3) ──────────────────────
        # Inject Gaussian noise into REPORTED position only.  Ground-truth
        # self.px / self.py remain untouched for physics & reward.
        noisy_x = self.px + random.gauss(0.0, 0.1)
        noisy_y = self.py + random.gauss(0.0, 0.1)

        return Observation(
            rover_position            = Vec3(x=noisy_x, y=noisy_y, z=t_height),
            rover_heading             = self.heading,
            rover_velocity            = Vec3(
                x=self.speed * math.cos(self.heading),
                y=self.speed * math.sin(self.heading),
                z=0.0,
            ),
            target_position           = Vec3(x=wx, y=wy, z=0.0),
            target_relative           = Vec3(x=dx, y=dy, z=0.0),
            target_distance           = self._clamp(dist, 0.0, 1414.0),
            waypoints_remaining       = self._clamp(waypts_rem, 0, 3),
            obstacle_map              = obs_map,
            obstacle_count            = len(nearest_raw),
            nearest_obstacle_distance = self._clamp(nearest_dist, 0.0, SENSOR_RANGE),
            battery_level             = self.battery,
            battery_drain_rate        = self._clamp(current_drain, 0.0, 1.0),
            terrain_type              = t_type,
            terrain_slope             = [slope_x, slope_y],
            steps_taken               = float(self.steps),
            steps_remaining_norm      = self._clamp(1.0 - self.steps / self.max_steps, 0.0, 1.0),
        )

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def get_obs(self) -> Observation:
        """Return current observation without advancing the simulation."""
        return self._build_obs()

    def step(self, action: Action) -> StepResponse:
        """
        Advance one timestep. Physics order:
          1. Kinematics  (heading -> speed -> position)
          2. Battery     (drain + optional regen)
          3. Collision   (penalty + speed zero)
          4. Waypoints   (check arrival)
          5. Termination (done / truncated flags)
          6. Reward
          7. Observation snapshot
        """
        if self.done or self.truncated:
            raise RuntimeError("Episode over — call /reset.")

        self.steps += 1

        # ── Sim-to-Real: Servo Rate Limiting (Feature 2) ───────────────
        # Clamp steering delta to ±0.5 per step so the LLM cannot command
        # instantaneous full-lock turns (mimics real servo slew rate).
        max_delta = 0.5
        clamped_steering = self._clamp(
            action.steering,
            self.previous_steering - max_delta,
            self.previous_steering + max_delta,
        )
        # Build a rate-limited copy of the action for kinematics
        action = Action(
            thrust=action.thrust,
            steering=clamped_steering,
            brake=action.brake,
            vertical_thruster=action.vertical_thruster,
        )

        # Snapshot distance BEFORE kinematics so potential-based shaping
        # can compute Δd = prev_dist − curr_dist for this step.
        prev_dist = self._prev_distance

        self._apply_kinematics(action)

        # Update previous_steering AFTER kinematics for next step's clamp
        self.previous_steering = clamped_steering
        drain        = self._update_battery(action)
        collided, nd = self._check_collision()
        wp_hit       = self._check_waypoints()

        # Track closest approach (drives partial-progress score)
        wp = self.active_waypoint or self.waypoint_list[-1]
        current_dist = math.hypot(wp[0] - self.px, wp[1] - self.py)
        if current_dist < self.min_distance:
            self.min_distance = current_dist

        # Update _prev_distance for the NEXT step's shaping computation
        self._prev_distance = current_dist

        all_done     = self.waypoints_hit == len(self.waypoint_list)
        batt_dead    = self.battery <= 0.0
        self.done      = all_done or batt_dead
        self.truncated = (not self.done) and (self.steps >= self.max_steps)

        # Termination reason — surfaced in info so baseline.py can pass it
        # directly to /grader without maintaining separate state.
        if all_done:
            termination_reason = "waypoint_reached"
        elif batt_dead:
            termination_reason = "battery_dead"
        elif self.truncated:
            termination_reason = "max_steps"
        else:
            termination_reason = "unknown"

        reward = self._compute_reward(wp_hit, collided, drain, prev_dist)
        self.total_reward += reward

        obs = self._build_obs()
        info: dict[str, Any] = {
            "steps":               self.steps,
            "max_steps":           self.max_steps,
            "waypoints_hit":       self.waypoints_hit,
            "total_waypoints":     len(self.waypoint_list),
            "collision":           collided,
            "battery":             round(self.battery, 4),
            "nearest_obstacle":    round(nd, 2),
            "total_reward":        round(self.total_reward, 4),
            # Grader telemetry — pass these directly to /grader at episode end
            "termination_reason":  termination_reason,
            "initial_distance":    round(self.initial_distance, 4),
            "min_distance":        round(self.min_distance, 4),
            "collision_count":     self.collision_count,
        }
        return StepResponse(obs=obs, reward=reward,
                            done=self.done, truncated=self.truncated, info=info)


# =============================================================================
# Episode factory
# =============================================================================

def _make_sim(task_id: str, seed: int | None) -> RoverSim:
    """
    Build a fully initialised RoverSim.

    Spawn rules
    -----------
    - Rover always starts at (0, 0), heading = 0 rad (east), battery = 1.0.
    - Waypoints placed at random angles, distances in
      [world_radius*0.3, world_radius*0.9].  Successive waypoints must be
      >= 40 m apart and >= 5 m from any obstacle centre.
    - Obstacle exclusion zone of 15 m around (0, 0) ensures a clear launch pad.
    """
    cfg = TASK_CONFIG[task_id]
    rng = random.Random(seed)

    terrain   = TerrainGrid(
        rng=random.Random(rng.randint(0, 2**31)),
        profile=cfg["terrain_profile"],
    )
    # ------------------------------------------------------------------
    # Step 1: Waypoint placement
    # Must happen BEFORE obstacle placement so the medium task can aim
    # its crater ring at the midpoint of the rover → waypoint line.
    # ------------------------------------------------------------------
    wp_rng  = random.Random(rng.randint(0, 2**31))
    waypoints: list[tuple[float, float]] = []
    prev_x, prev_y = 0.0, 0.0

    for _ in range(500):
        if len(waypoints) >= cfg["waypoints"]:
            break
        angle  = wp_rng.uniform(0, 2 * math.pi)
        r      = wp_rng.uniform(cfg["world_radius"] * 0.35, cfg["world_radius"] * 0.85)
        wx, wy = r * math.cos(angle), r * math.sin(angle)

        # Enforce minimum spacing between consecutive waypoints
        if math.hypot(wx - prev_x, wy - prev_y) < 30.0:
            continue

        waypoints.append((wx, wy))
        prev_x, prev_y = wx, wy

    if not waypoints:
        # Deterministic fallback: place waypoint due east at 60 % of world radius
        waypoints = [(cfg["world_radius"] * 0.6, 0.0)]

    # ------------------------------------------------------------------
    # Step 2: Obstacle field
    #
    #   easy / hard — ObstacleField.generate() with density=0.0 → empty
    #   medium      — deterministic crater ring centred at the midpoint of
    #                 the rover → first-waypoint straight line, with two
    #                 symmetric gaps that force a detour
    # ------------------------------------------------------------------
    if cfg.get("crater_obstacle"):
        wx0, wy0 = waypoints[0]
        obstacles = ObstacleField.place_crater_ring(wx0, wy0)
    else:
        obstacles = ObstacleField.generate(
            rng=random.Random(rng.randint(0, 2**31)),
            density=cfg["obstacle_density"],
            world_radius=cfg["world_radius"],
        )

    # ------------------------------------------------------------------
    # Step 3: Grading telemetry
    # initial_distance is the straight-line spawn → first waypoint distance.
    # min_distance starts equal to initial_distance and falls each step;
    # it drives partial-progress scoring in _compute_score.
    # ------------------------------------------------------------------
    initial_dist = math.hypot(waypoints[0][0], waypoints[0][1])

    # ── Sim-to-Real: Domain Randomization (Feature 1) ──────────────────
    # Each episode gets a random physics modifier ∈ [0.9, 1.1] that
    # slightly scales speed, simulating terrain friction / gravity jitter.
    physics_mod = rng.uniform(0.9, 1.1)

    return RoverSim(
        task_id=task_id,
        max_steps=cfg["max_steps"],
        drain_mult=cfg["battery_drain_mult"],
        terrain=terrain,
        obstacles=obstacles,
        waypoint_list=waypoints,
        px=0.0, py=0.0, heading=0.0, speed=0.0,
        battery=cfg.get('starting_battery', 1.0), steps=0,
        done=False, truncated=False, waypoints_hit=0,
        initial_distance=initial_dist,
        min_distance=initial_dist,
        _prev_distance=initial_dist,
        collision_count=0,
        physics_modifier=physics_mod,       # Feature 1: domain randomization
        previous_steering=0.0,              # Feature 2: servo rate limiter init
    )


# =============================================================================
# In-memory episode store
# =============================================================================

class SimulationStore:
    def __init__(self) -> None:
        self._sims: dict[str, RoverSim] = {}

    def new(self, task_id: str, seed: int | None) -> tuple[str, RoverSim]:
        eid = str(uuid.uuid4())
        self._sims[eid] = _make_sim(task_id, seed)
        return eid, self._sims[eid]

    def get(self, episode_id: str) -> RoverSim:
        sim = self._sims.get(episode_id)
        if sim is None:
            raise HTTPException(404, f"Episode '{episode_id}' not found. Call /reset first.")
        return sim


_store = SimulationStore()


# =============================================================================
# Grading helpers
# =============================================================================

def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _proximity_progress(initial: float, closest: float) -> float:
    """
    Linear partial-progress metric.

    Maps the rover's closest recorded approach to the waypoint onto [0.0, 1.0]
    using a straight linear scale against the initial spawn→waypoint distance:

        proximity = 1.0 - (min_distance_achieved / initial_distance)

    Calibration table (matches the "gets N% of the way → returns N/100" contract):

        Progress   min_distance_achieved   proximity returned
        ────────   ─────────────────────   ─────────────────
          0 %       == initial_distance        0.00
         30 %       0.70 × initial             0.30
         50 %       0.50 × initial             0.50
         70 %       0.30 × initial             0.70  ← key example from spec
         90 %       0.10 × initial             0.90
        100 %       ≤ WAYPOINT_RADIUS          1.00  (hard override in _compute_score)

    No smoothing is applied here — task formulas apply their own weights.
    A previous sqrt-smoothed version returned ≈0.837 for 70% progress, which
    violated the partial-progress contract. This version is exact.
    """
    if initial <= 0.0:
        return 1.0
    progress = 1.0 - _clamp01(closest / initial)
    return round(progress, 6)


# ── Verdict labels ─────────────────────────────────────────────────────────

_VERDICT_WIN              = "WIN"
_VERDICT_WIN_COLLISIONS   = "WIN_WITH_COLLISIONS"
_VERDICT_PARTIAL          = "PARTIAL_PROGRESS"
_VERDICT_COLLISION_LOSS   = "COLLISION_LOSS"
_VERDICT_BATTERY_DEAD     = "BATTERY_DEAD"
_VERDICT_TIMEOUT          = "TIMEOUT"


def _compute_score(req: GraderRequest) -> tuple[float, str, str, dict[str, float]]:
    """
    Task-specific grading with explicit win / partial-progress / loss paths.

    Returns
    -------
    (score, verdict, rationale, breakdown)

    score    : float in [0.0, 1.0]
    verdict  : one of the _VERDICT_* constants
    rationale: one-sentence human-readable explanation
    breakdown: per-component float dict (keys vary by task)

    ══════════════════════════════════════════════════════════════════════
    Shared building blocks
    ══════════════════════════════════════════════════════════════════════

    proximity_progress  (linear, [0.0, 1.0])
        = 1.0 - (min_distance_achieved / initial_distance)
        Exactly 0.70 when the rover closed 70 % of the gap.
        Overridden to 1.0 on confirmed arrival (waypoints_reached >= total).

    step_efficiency  ([0.0, 1.0])
        = 1.0 - (steps_taken / max_steps)
        0.0 when the full budget was consumed; 1.0 if done in step 1.

    battery_efficiency  ([0.0, 1.0])  — hard task only
        = battery_remaining / starting_battery
        Normalised against the task's starting charge (0.35) so the hard task
        doesn't punish the rover for starting with less than a full battery.

    ══════════════════════════════════════════════════════════════════════
    EASY  —  "Flat Plains Transit"
    ══════════════════════════════════════════════════════════════════════
    Challenge : pure navigation, no obstacles, no battery constraint.
    Formula   : proximity * 0.85 + step_efficiency * 0.15

    Condition paths
    ───────────────
    WIN           waypoints_reached >= total
                  score = 1.0 * 0.85 + step_eff * 0.15
                  Maximum 1.00 (arrived in first step), minimum 0.85 (full budget).

    PARTIAL       didn't arrive but made measurable progress (proximity > 0)
                  score = proximity * 0.85 + step_eff * 0.15
                  "70% of the way, half budget used" → 0.70*0.85 + 0.50*0.15 = 0.67

    TIMEOUT       step budget exhausted without arrival, no progress
                  score = 0.0 * 0.85 + 0.0 * 0.15 = 0.00

    Example scores
    ──────────────
    Arrive, use 50 % budget  : 0.85 + 0.075 = 0.925
    Arrive, use 100 % budget : 0.85 + 0.000 = 0.850
    70 % there, 50 % budget  : 0.595 + 0.075 = 0.670
    30 % there, full budget  : 0.255 + 0.000 = 0.255

    ══════════════════════════════════════════════════════════════════════
    MEDIUM  —  "Crater Avoidance"
    ══════════════════════════════════════════════════════════════════════
    Challenge : navigate around a static crater ring; each contact is penalised.
    Formula   : proximity * 0.75 + step_efficiency * 0.25 − collision_penalty
    where     : collision_penalty = min(collision_count * 0.06, 0.40)

    Condition paths
    ───────────────
    WIN                   arrived, zero collisions
                          score = 0.75 + step_eff * 0.25

    WIN_WITH_COLLISIONS   arrived, but hit the ring
                          score = (0.75 + step_eff * 0.25) − penalty
                          Capped at 0 so the rover can't score negative.

    PARTIAL_PROGRESS      didn't arrive, mild collisions (penalty < proximity component)
                          score = proximity * 0.75 + step_eff * 0.25 − penalty

    COLLISION_LOSS        so many collisions that score floor is 0
                          verdict = COLLISION_LOSS even if proximity > 0

    BATTERY_DEAD          battery hit 0 mid-episode (shouldn't happen on medium;
                          drain_mult=1.0, but handled defensively)

    Example scores
    ──────────────
    Arrive, 0 collisions, 50 % budget : 0.75 + 0.125 = 0.875
    Arrive, 3 collisions, 50 % budget : 0.875 − 0.18 = 0.695
    70 % there, 1 collision, 50 %     : 0.525 + 0.125 − 0.06 = 0.590
    Stuck in ring, 8 collisions       : any_proximity − 0.40 → likely ≤ 0.00

    ══════════════════════════════════════════════════════════════════════
    HARD  —  "Battery Sprint"
    ══════════════════════════════════════════════════════════════════════
    Challenge : 35 % starting battery + ×4 drain; any detour is fatal.
    Formula   : proximity * 0.65 + battery_efficiency * 0.35
    where     : battery_efficiency = battery_remaining / HARD_START_BATTERY (0.35)

    Normalising against starting_battery (0.35) means a rover that arrives
    having used exactly half its charge gets battery_efficiency = 0.50, not
    (0.175/1.0) = 0.175.  This makes the battery component human-readable.

    Condition paths
    ───────────────
    WIN             arrived (proximity = 1.0)
                    score = 0.65 + batt_eff * 0.35
                    Maximum 1.00 (arrive with full starting charge, impossible
                    in practice but mathematically correct ceiling).

    BATTERY_DEAD    battery hit 0 before arrival
                    score = proximity * 0.65 + 0.0 * 0.35
                    "70 % there, battery dead" → 0.70 * 0.65 = 0.455
                    This is the primary failure mode for this task.

    TIMEOUT         ran out of steps — scores proximity + whatever battery remains.

    Example scores
    ──────────────
    Arrive, 50 % start-bat left   : 0.65 + (0.175/0.35)*0.35 = 0.65 + 0.175 = 0.825
    Arrive, battery = 0 on arrival: 0.65 + 0.00 = 0.650
    70 % there, battery dead      : 0.455 + 0.00 = 0.455
    0 % progress, battery dead    : 0.000 + 0.00 = 0.000
    """

    HARD_START_BATTERY = TASK_CONFIG["hard"].get("starting_battery", 0.35)

    # ── Shared inputs ──────────────────────────────────────────────────────
    arrived  = req.waypoints_reached >= req.total_waypoints
    proximity = _proximity_progress(req.initial_distance, req.min_distance_achieved)

    # Confirmed arrival overrides floating-point near-misses
    if arrived:
        proximity = 1.0

    step_eff = _clamp01(1.0 - req.steps_taken / req.max_steps)
    batt     = _clamp01(req.battery_remaining)
    reason   = req.termination_reason

    # ── EASY ───────────────────────────────────────────────────────────────
    if req.task_id == "easy":
        p_score = round(proximity * 0.85, 4)
        e_score = round(step_eff  * 0.15, 4)
        raw     = p_score + e_score

        if arrived:
            verdict   = _VERDICT_WIN
            rationale = (
                f"Waypoint reached in {req.steps_taken}/{req.max_steps} steps "
                f"(efficiency {step_eff:.0%})."
            )
        elif proximity > 0.0:
            verdict   = _VERDICT_PARTIAL
            rationale = (
                f"Rover closed {proximity:.0%} of the gap "
                f"({req.min_distance_achieved:.1f} m from waypoint) "
                f"before {reason.replace('_', ' ')}."
            )
        else:
            verdict   = _VERDICT_TIMEOUT
            rationale = "No progress recorded — rover did not move toward the waypoint."

        score     = round(_clamp01(raw), 4)
        breakdown = {
            "proximity_component":  p_score,
            "efficiency_component": e_score,
            "total":                score,
        }
        return score, verdict, rationale, proximity, breakdown

    # ── MEDIUM ─────────────────────────────────────────────────────────────
    elif req.task_id == "medium":
        collision_penalty = round(min(req.collision_count * 0.06, 0.40), 4)
        p_score           = round(proximity * 0.75, 4)
        e_score           = round(step_eff  * 0.25, 4)
        raw               = p_score + e_score - collision_penalty

        if arrived and req.collision_count == 0:
            verdict   = _VERDICT_WIN
            rationale = (
                f"Waypoint reached with zero collisions in "
                f"{req.steps_taken}/{req.max_steps} steps."
            )
        elif arrived and req.collision_count > 0:
            verdict   = _VERDICT_WIN_COLLISIONS
            rationale = (
                f"Waypoint reached but {req.collision_count} collision(s) "
                f"applied a -{collision_penalty:.2f} penalty."
            )
        elif raw <= 0.0:
            verdict   = _VERDICT_COLLISION_LOSS
            rationale = (
                f"{req.collision_count} collision(s) (penalty {collision_penalty:.2f}) "
                f"erased all proximity/efficiency gains."
            )
        elif reason == "battery_dead":
            verdict   = _VERDICT_BATTERY_DEAD
            rationale = (
                f"Battery depleted at {proximity:.0%} progress; "
                f"{req.collision_count} collision(s) added penalty."
            )
        else:
            verdict   = _VERDICT_PARTIAL
            rationale = (
                f"Rover closed {proximity:.0%} of the gap "
                f"({req.collision_count} collision(s), penalty -{collision_penalty:.2f})."
            )

        score     = round(_clamp01(raw), 4)
        breakdown = {
            "proximity_component":   p_score,
            "efficiency_component":  e_score,
            "collision_penalty":    -collision_penalty,
            "total":                 score,
        }
        return score, verdict, rationale, proximity, breakdown

    # ── HARD ───────────────────────────────────────────────────────────────
    else:
        # Normalise battery against starting charge so the component is
        # human-readable: 1.0 = no battery consumed, 0.0 = fully depleted.
        batt_eff  = round(_clamp01(batt / HARD_START_BATTERY), 6)
        p_score   = round(proximity * 0.65, 4)
        b_score   = round(batt_eff  * 0.35, 4)
        raw       = p_score + b_score

        if arrived:
            verdict   = _VERDICT_WIN
            rationale = (
                f"Waypoint reached with {batt:.1%} battery remaining "
                f"({batt_eff:.0%} of starting charge conserved)."
            )
        elif reason == "battery_dead":
            verdict   = _VERDICT_BATTERY_DEAD
            rationale = (
                f"Battery exhausted at {proximity:.0%} progress "
                f"({req.min_distance_achieved:.1f} m from waypoint)."
            )
        elif proximity > 0.0:
            verdict   = _VERDICT_PARTIAL
            rationale = (
                f"Rover closed {proximity:.0%} of the gap; "
                f"{batt:.1%} battery left at {reason.replace('_', ' ')}."
            )
        else:
            verdict   = _VERDICT_TIMEOUT
            rationale = "No proximity progress before step budget exhausted."

        score     = round(_clamp01(raw), 4)
        breakdown = {
            "proximity_component":         p_score,
            "battery_efficiency_component": b_score,
            "battery_remaining_raw":       round(batt, 4),
            "battery_normalised":          round(batt_eff, 4),
            "total":                       score,
        }
        return score, verdict, rationale, proximity, breakdown


# =============================================================================
# FastAPI application
# =============================================================================

app = FastAPI(
    title="Planetary Rover Navigation Simulator",
    description="OpenEnv-compliant RL environment — Meta PyTorch Hackathon Round 1",
    version="1.0.0",
)


# Action schema is identical across all tasks — defined once, attached to every TaskMeta.
# Describes every field in the Action model: type, bounds, and semantics.
_ACTION_SCHEMA: dict[str, Any] = {
    "thrust": {
        "type":        "Box",
        "dtype":       "float32",
        "low":         0.0,
        "high":        1.0,
        "description": "Forward drive intensity. 0.0 = stopped, 1.0 = full throttle.",
    },
    "steering": {
        "type":        "Box",
        "dtype":       "float32",
        "low":         -1.0,
        "high":        1.0,
        "description": (
            "Yaw-rate command. -1.0 = hard left, 0.0 = straight, 1.0 = hard right. "
            "Effective yaw rate scales with current thrust."
        ),
    },
    "brake": {
        "type":        "Discrete",
        "dtype":       "int32",
        "n":           2,
        "description": (
            "Binary brake flag. 1 = apply regenerative braking: "
            "halves speed and recovers a small amount of battery energy. "
            "0 = coast or drive normally."
        ),
    },
    "vertical_thruster": {
        "type":        "Box",
        "dtype":       "float32",
        "low":         -0.2,
        "high":        0.2,
        "description": (
            "Small vertical adjustment thruster. Only has physical effect "
            "on crater_floor / crater_rim terrain. Ignored (zero battery cost) "
            "on flat terrain."
        ),
    },
}


@app.get("/tasks", response_model=list[TaskMeta], tags=["OpenEnv"])
def get_tasks() -> list[TaskMeta]:
    """
    Return metadata for all three tasks, including the full action schema.

    Each TaskMeta includes:
      • Core task parameters (difficulty, max_steps, terrain, obstacles, battery)
      • A human-readable description of the challenge
      • action_schema: the complete typed action space the agent must send
        to /step — field names, types, bounds, and semantics

    Intended use
    ────────────
    Agents and baselines call this endpoint once at startup to:
      1. Discover available task IDs for /reset
      2. Understand the observation/action contract before building a policy
      3. Auto-generate random-action baselines from the declared bounds
    """
    tasks = []
    for tid, cfg in TASK_CONFIG.items():
        # Representative full-thrust drain scaled by task multiplier
        drain_rate = round((DRAIN_BASE + DRAIN_PER_THRUST) * cfg["battery_drain_mult"], 6)
        tasks.append(TaskMeta(
            id                 = tid,
            display_name       = cfg["display_name"],
            description        = cfg["description"],
            difficulty         = cfg["difficulty"],
            max_steps          = cfg["max_steps"],
            waypoints          = cfg["waypoints"],
            terrain_profile    = cfg["terrain_profile"],
            obstacle_density   = cfg["obstacle_density"],
            battery_drain_rate = drain_rate,
            starting_battery   = cfg.get("starting_battery", 1.0),
            target_score       = 1.0,
            scoring_formula    = cfg.get("scoring_formula", ""),
            hints              = cfg.get("hints", []),
            action_schema      = _ACTION_SCHEMA,
        ))
    return tasks


@app.post("/reset", response_model=ResetResponse, tags=["OpenEnv"])
def reset(req: ResetRequest | None = None) -> ResetResponse:
    """
    Initialise a new episode.

    Rover spawns at (0, 0), heading east (0 rad), battery = 100%.
    Waypoints are seeded from task config + optional RNG seed.
    Returns initial Observation and episode_id for all subsequent calls.
    """
    # If the bot sends a null/empty request, manually create a default one
    if req is None:
        req = ResetRequest(task_id="easy", seed=None)

    if req.task_id not in TASK_CONFIG:
        raise HTTPException(422, f"task_id must be one of {sorted(TASK_CONFIG)}")
    eid, sim = _store.new(req.task_id, req.seed)
    return ResetResponse(obs=sim.get_obs(), episode_id=eid, task_id=req.task_id)


@app.get("/state", response_model=Observation, tags=["OpenEnv"])
def state(episode_id: str) -> Observation:
    """
    Return the current Observation without advancing the simulation.
    Safe to call at any point during an episode.
    Query param: ?episode_id=<uuid>
    """
    return _store.get(episode_id).get_obs()


@app.post("/step", response_model=StepResponse, tags=["OpenEnv"])
def step(episode_id: str, action: Action) -> StepResponse:
    """
    Apply one action and advance the simulation by one timestep (dt = 1 s).

    Physics pipeline per step:
      1. Kinematics  — yaw-rate steering, thrust->speed, Euler position update
      2. Battery     — base + terrain + thrust cost, regen on brake
      3. Collision   — obstacle proximity; zeroes speed on contact
      4. Waypoints   — checks 2 m arrival radius; increments counter
      5. Termination — done = all waypoints reached OR battery = 0;
                       truncated = max_steps exceeded
      6. Reward      — step penalty, waypoint bonus, collision/battery penalties
    """
    sim = _store.get(episode_id)
    if sim.done or sim.truncated:
        raise HTTPException(409, "Episode finished. Call /reset.")
    try:
        return sim.step(action)
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.get("/baseline", response_model=BaselineResponse, tags=["OpenEnv"])
def baseline() -> BaselineResponse:
    """Machine-readable environment identity and space declarations."""
    obs_fields = [
        SpaceField(name="rover_position",           type="Box",      shape=[3],   dtype="float32",
                   low=[-500,-500,-50], high=[500,500,50],     description="[x,y,z] rover position (m)"),
        SpaceField(name="rover_heading",             type="Box",      shape=[1],   dtype="float32",
                   low=[-3.14159], high=[3.14159],              description="Yaw (rad)"),
        SpaceField(name="rover_velocity",            type="Box",      shape=[3],   dtype="float32",
                   low=[-5,-5,-2], high=[5,5,2],                description="[vx,vy,vz] m/s"),
        SpaceField(name="target_position",           type="Box",      shape=[3],   dtype="float32",
                   low=[-500,-500,-50], high=[500,500,50],     description="Active waypoint (m)"),
        SpaceField(name="target_relative",           type="Box",      shape=[3],   dtype="float32",
                   low=[-1000,-1000,-100], high=[1000,1000,100], description="Rover->waypoint vector"),
        SpaceField(name="target_distance",           type="Box",      shape=[1],   dtype="float32",
                   low=[0.0], high=[1414.0],                    description="Distance to waypoint (m)"),
        SpaceField(name="waypoints_remaining",       type="Discrete", shape=None,  dtype="int32",
                   low=0, high=3,                               description="Unvisited waypoints"),
        SpaceField(name="obstacle_map",              type="Box",      shape=[8,3], dtype="float32",
                   low=-1.0, high=1.0,                          description="8 nearest [dx,dy,dist]_norm"),
        SpaceField(name="obstacle_count",            type="Discrete", shape=None,  dtype="int32",
                   low=0, high=8,                               description="Obstacles in sensor range"),
        SpaceField(name="nearest_obstacle_distance", type="Box",      shape=[1],   dtype="float32",
                   low=[0.0], high=[50.0],                      description="Closest obstacle (m)"),
        SpaceField(name="battery_level",             type="Box",      shape=[1],   dtype="float32",
                   low=[0.0], high=[1.0],                       description="Battery [0=dead,1=full]"),
        SpaceField(name="battery_drain_rate",        type="Box",      shape=[1],   dtype="float32",
                   low=[0.0], high=[1.0],                       description="Drain per step / capacity"),
        SpaceField(name="terrain_type",              type="Discrete", shape=None,  dtype="int32",
                   low=0, high=3,                               description="0=flat 1=rocky 2=floor 3=rim"),
        SpaceField(name="terrain_slope",             type="Box",      shape=[2],   dtype="float32",
                   low=[-1.0,-1.0], high=[1.0,1.0],             description="[slope_x,slope_y]"),
        SpaceField(name="steps_taken",               type="Box",      shape=[1],   dtype="float32",
                   low=[0.0], high=[500.0],                     description="Steps elapsed"),
        SpaceField(name="steps_remaining_norm",      type="Box",      shape=[1],   dtype="float32",
                   low=[0.0], high=[1.0],                       description="Remaining budget [0,1]"),
    ]
    act_fields = [
        SpaceField(name="thrust",            type="Box",      shape=[1],  dtype="float32",
                   low=[0.0],  high=[1.0],   description="Forward drive intensity"),
        SpaceField(name="steering",          type="Box",      shape=[1],  dtype="float32",
                   low=[-1.0], high=[1.0],   description="Yaw rate command"),
        SpaceField(name="brake",             type="Discrete", shape=None, dtype="int32",
                   low=0, high=1,            description="Regen braking flag"),
        SpaceField(name="vertical_thruster", type="Box",      shape=[1],  dtype="float32",
                   low=[-0.2], high=[0.2],   description="Vertical adjust (crater only)"),
    ]
    return BaselineResponse(
        name="planetary-rover-navigation", version="1.0.0",
        description="Planetary rover nav sim — reach waypoints, manage battery, avoid obstacles.",
        observation_space=obs_fields, action_space=act_fields,
        tasks=["easy", "medium", "hard"],
    )


@app.post("/grader", response_model=GraderResponse, tags=["OpenEnv"])
def grader(req: GraderRequest) -> GraderResponse:
    """
    Score a completed episode and return a float in [0.0, 1.0].

    Accepts a GraderRequest containing the trajectory summary produced by the
    final StepResponse `info` dict.  All required fields are emitted by the
    simulation automatically — no client-side bookkeeping needed.

    Scoring is task-specific:

      EASY   : proximity * 0.85 + step_efficiency * 0.15
      MEDIUM : proximity * 0.75 + step_efficiency * 0.25 − collision_penalty
      HARD   : proximity * 0.65 + battery_efficiency * 0.35

    where proximity is a LINEAR metric: exactly 0.70 when the rover closed
    70 % of the spawn→waypoint gap, exactly 1.0 on confirmed arrival.

    The response includes:
      • score             : the final [0.0, 1.0] float
      • verdict           : WIN | WIN_WITH_COLLISIONS | PARTIAL_PROGRESS |
                            COLLISION_LOSS | BATTERY_DEAD | TIMEOUT
      • proximity_progress: the raw linear proximity value (task-weight-free)
      • score_rationale   : one-sentence plain-English explanation
      • breakdown         : per-component floats (keys vary by task)

    Validation
    ----------
    • task_id must be "easy", "medium", or "hard"
    • min_distance_achieved must be <= initial_distance
    • steps_taken must be <= max_steps
    """
    if req.task_id not in TASK_CONFIG:
        raise HTTPException(422, f"task_id must be one of {sorted(TASK_CONFIG)}")

    if req.min_distance_achieved > req.initial_distance + 1e-6:
        raise HTTPException(
            422,
            f"min_distance_achieved ({req.min_distance_achieved:.2f}) cannot exceed "
            f"initial_distance ({req.initial_distance:.2f})."
        )

    if req.steps_taken > req.max_steps:
        raise HTTPException(
            422,
            f"steps_taken ({req.steps_taken}) cannot exceed max_steps ({req.max_steps})."
        )

    score, verdict, rationale, proximity, breakdown = _compute_score(req)

    return GraderResponse(
        episode_id         = req.episode_id,
        task_id            = req.task_id,
        score              = score,
        verdict            = verdict,
        proximity_progress = round(proximity, 4),
        score_rationale    = rationale,
        breakdown          = breakdown,
    )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
