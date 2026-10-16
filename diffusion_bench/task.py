"""Planar point-mass task used by the behavior experiments.

Two demonstration corridors leave the origin and meet at a goal. Their average
passes through a disk obstacle. A 6 cm lateral jump is an instantaneous state
offset in meters. This is not a rigid-body simulator.
"""

from __future__ import annotations

import numpy as np

START = np.array([0.0, 0.0], dtype=np.float64)
GOAL = np.array([0.24, 0.0], dtype=np.float64)
OBSTACLE_CENTER = np.array([0.12, 0.0], dtype=np.float64)
OBSTACLE_RADIUS = 0.04
GOAL_TOL = 0.02
N_ACTIONS = 16
PRED_HORIZON = 16
ACTION_DIM = 2
OBS_DIM = 7
AMP = 0.09
PERTURBATION_M = 0.06
DISTURB_AFTER_STEPS = 4


def observation(xy: np.ndarray) -> np.ndarray:
    point = np.asarray(xy, dtype=np.float64)
    return np.array(
        [
            point[0],
            point[1],
            GOAL[0],
            GOAL[1],
            OBSTACLE_CENTER[0],
            OBSTACLE_CENTER[1],
            OBSTACLE_RADIUS,
        ],
        dtype=np.float32,
    )


def nominal_positions(sign: float, amp: float = AMP, n_actions: int = N_ACTIONS) -> np.ndarray:
    """Sine corridor from START to GOAL. `sign` selects the +y or -y side."""
    fraction = np.linspace(0.0, 1.0, n_actions + 1)
    pos = np.zeros((n_actions + 1, 2), dtype=np.float64)
    pos[:, 0] = START[0] + (GOAL[0] - START[0]) * fraction
    pos[:, 1] = sign * amp * np.sin(np.pi * fraction)
    return pos


def bridge_positions(start: np.ndarray, sign: float, amp: float = AMP, n_actions: int = N_ACTIONS) -> np.ndarray:
    """Fixed-speed path from `start` to GOAL, then a hold.

    X advances at the same step used by the origin corridor, so executing the
    first few offsets and replanning does not restart a stretched 16-step ramp.
    Y is a straight lerp when that line clears the disk, otherwise a sine bow
    on `sign`'s side. From the origin the bow is the bimodal corridor.
    """
    start = np.asarray(start, dtype=np.float64)
    x_step = (GOAL[0] - START[0]) / n_actions
    remaining = max(GOAL[0] - float(start[0]), 0.0)
    n_move = int(np.ceil(remaining / x_step - 1e-9)) if x_step > 0 else 0
    n_move = min(max(n_move, 0), n_actions)
    pos = np.tile(start, (n_actions + 1, 1))
    if n_move == 0:
        pos[:] = GOAL
        pos[0] = start
        return pos

    fraction = np.zeros(n_actions + 1, dtype=np.float64)
    fraction[1 : n_move + 1] = np.arange(1, n_move + 1) / n_move
    fraction[n_move:] = 1.0
    pos[:, 0] = start[0] + (GOAL[0] - start[0]) * fraction
    lerp_y = start[1] * (1.0 - fraction)
    bow_y = sign * amp * np.sin(np.pi * fraction)
    straight = pos.copy()
    straight[:, 1] = lerp_y
    if path_hits_disk(straight):
        if sign >= 0.0:
            pos[:, 1] = np.maximum(lerp_y, bow_y)
        else:
            pos[:, 1] = np.minimum(lerp_y, bow_y)
    else:
        pos[:, 1] = lerp_y
    pos[0] = start
    pos[n_move:] = GOAL
    return pos


def segment_hits_disk(p0: np.ndarray, p1: np.ndarray, center: np.ndarray = OBSTACLE_CENTER, radius: float = OBSTACLE_RADIUS) -> bool:
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    delta = p1 - p0
    offset = p0 - center
    a = float(np.dot(delta, delta))
    c = float(np.dot(offset, offset) - radius * radius)
    if a < 1e-12:
        return c <= 0.0
    b = float(2.0 * np.dot(offset, delta))
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    root = disc ** 0.5
    t1 = (-b - root) / (2.0 * a)
    t2 = (-b + root) / (2.0 * a)
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0)


def path_hits_disk(points: np.ndarray) -> bool:
    pts = np.asarray(points, dtype=np.float64)
    if np.any(np.linalg.norm(pts - OBSTACLE_CENTER, axis=1) < OBSTACLE_RADIUS):
        return True
    return any(segment_hits_disk(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def min_clearance(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    distances = np.linalg.norm(pts - OBSTACLE_CENTER, axis=1)
    return float(distances.min() - OBSTACLE_RADIUS)


def chunks_from_positions(positions: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """One chunk of cumulative offsets from the first point to each later point.

    offsets[k] = position[k+1] - position[0]. The last entry is the goal offset
    when the path ends at GOAL. Open-loop execution differences these offsets, so
    a mid-chunk state jump is not cancelled.
    """
    offsets = (positions[1:] - positions[0]).astype(np.float32)
    if offsets.shape[0] != PRED_HORIZON:
        chunk = np.zeros((PRED_HORIZON, ACTION_DIM), dtype=np.float32)
        n = min(PRED_HORIZON, offsets.shape[0])
        chunk[:n] = offsets[:n]
        offsets = chunk
    return [(observation(positions[0]), offsets)]


def _append_bridge(obs_rows: list[np.ndarray], act_rows: list[np.ndarray], start: np.ndarray, sign: float, amp: float) -> bool:
    if np.linalg.norm(start - OBSTACLE_CENTER) < OBSTACLE_RADIUS + 0.008:
        return False
    positions = bridge_positions(start, sign=sign, amp=amp)
    if path_hits_disk(positions):
        return False
    obs, act = chunks_from_positions(positions)[0]
    obs_rows.append(obs)
    act_rows.append(act)
    return True


def build_dataset(rng: np.random.Generator, demos_per_mode: int = 400, recovery_attempts: int = 2400) -> dict:
    """Full-horizon bridges only.

    Every target chunk runs from the current point to the goal in 16 steps.
    Zero-padded tails are omitted: they ask the conv net for a shape it then
    mixes into the origin chunk. The origin is bimodal. Off-center starts use
    the nearer side, with a lateral shift up to 8 cm so a 6 cm jump is in view.
    """
    obs_rows: list[np.ndarray] = []
    act_rows: list[np.ndarray] = []
    nominal_kept = 0
    recovery_kept = 0

    for sign in (1.0, -1.0):
        for _ in range(demos_per_mode):
            amp = float(rng.uniform(0.088, 0.098))
            if _append_bridge(obs_rows, act_rows, START, sign, amp):
                nominal_kept += 1

    for _ in range(recovery_attempts):
        sign = float(rng.choice((-1.0, 1.0)))
        amp = float(rng.uniform(0.088, 0.098))
        base = nominal_positions(sign, amp=amp)
        index = int(rng.integers(0, base.shape[0] - 1))
        shift = float(rng.uniform(-0.08, 0.08))
        start = base[index] + np.array([0.0, shift])
        if start[0] > 0.18:
            continue
        side = sign if abs(float(start[1])) < 0.02 else float(np.sign(start[1]))
        if _append_bridge(obs_rows, act_rows, start, side, amp):
            recovery_kept += 1

    obs = np.stack(obs_rows).astype(np.float32)
    act = np.stack(act_rows).astype(np.float32)
    return {
        "obs": obs,
        "act": act,
        "nominal_kept": nominal_kept,
        "recovery_kept": recovery_kept,
        "samples": int(obs.shape[0]),
    }


def geometry_checks() -> dict:
    left = nominal_positions(1.0)
    right = nominal_positions(-1.0)
    mean = 0.5 * (left + right)
    left_hit = path_hits_disk(left)
    right_hit = path_hits_disk(right)
    mean_hit = path_hits_disk(mean)
    if left_hit or right_hit or not mean_hit:
        raise RuntimeError("corridor geometry no longer separates the mean path from the two modes")
    jumped = nominal_positions(1.0)
    jumped[DISTURB_AFTER_STEPS] = jumped[DISTURB_AFTER_STEPS] + np.array([0.0, PERTURBATION_M])
    return {
        "left_clearance_m": min_clearance(left),
        "right_clearance_m": min_clearance(right),
        "mean_clearance_m": min_clearance(mean),
        "mean_path_hits_obstacle": mean_hit,
        "origin_modes_clear": (not left_hit) and (not right_hit),
        "post_jump_index_clear": bool(np.linalg.norm(jumped[DISTURB_AFTER_STEPS] - OBSTACLE_CENTER) >= OBSTACLE_RADIUS),
    }
