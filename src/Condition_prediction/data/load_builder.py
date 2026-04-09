from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


DOF_MAP = {
    "fx": 0,
    "ux": 0,
    "fy": 1,
    "uy": 1,
    "rz": 2,
}


def _dof_index(node_id: int, dof: str, dofs_per_node: int) -> int:
    dof_lower = str(dof).lower()
    if dof_lower not in DOF_MAP:
        raise ValueError(f"Unsupported dof: {dof}")
    local_dof = DOF_MAP[dof_lower]
    if local_dof >= dofs_per_node:
        raise ValueError(
            f"dof '{dof}' requires local index {local_dof}, "
            f"but dofs_per_node={dofs_per_node}"
        )
    return int(node_id) * int(dofs_per_node) + local_dof


def build_deterministic_load_matrix(
    load_specs: List[Dict],
    num_nodes: int,
    dofs_per_node: int,
    dt: float,
    total_time: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build deterministic load matrix from a fixed load-case config.

    Returns:
        time_axis: shape=(num_steps,)
        load_matrix: shape=(num_steps, num_dofs)
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if total_time <= 0.0:
        raise ValueError("total_time must be positive.")

    num_steps = int(total_time / dt) + 1
    num_dofs = int(num_nodes) * int(dofs_per_node)
    t = np.arange(num_steps, dtype=float) * float(dt)
    load_matrix = np.zeros((num_steps, num_dofs), dtype=float)

    for spec in load_specs:
        node_id = int(spec["node_id"])
        dof_idx = _dof_index(node_id=node_id, dof=spec["dof"], dofs_per_node=dofs_per_node)
        pattern = str(spec.get("pattern", "harmonic")).lower()
        amplitude = float(spec.get("F0", 0.0))

        if pattern == "harmonic":
            freq = float(spec.get("freq", 1.0))
            phase = float(spec.get("phase", 0.0))
            offset = float(spec.get("offset", 0.0))
            duration = float(spec.get("duration", total_time))
            duration = min(duration, total_time)
            signal = amplitude * np.sin(2.0 * np.pi * freq * t + phase) + offset
            mask = t <= duration + 1e-12
            load_matrix[mask, dof_idx] += signal[mask]

        elif pattern == "pulse":
            t_start = float(spec.get("t_start", 0.0))
            t_end = float(spec.get("t_end", t_start))
            if t_end <= t_start:
                raise ValueError(f"pulse t_end must be larger than t_start: {spec}")
            mask = (t >= t_start) & (t < t_end)
            load_matrix[mask, dof_idx] += amplitude

        elif pattern == "half_sine":
            t_start = float(spec.get("t_start", 0.0))
            t_end = float(spec.get("t_end", t_start))
            if t_end <= t_start:
                raise ValueError(f"half_sine t_end must be larger than t_start: {spec}")
            mask = (t >= t_start) & (t < t_end)
            duration = t_end - t_start
            tau = t[mask] - t_start
            load_matrix[mask, dof_idx] += amplitude * np.sin(np.pi * tau / duration)

        elif pattern == "ramp":
            t_start = float(spec.get("t_start", 0.0))
            t_ramp = float(spec.get("t_ramp", 0.1))
            if t_ramp <= 0.0:
                raise ValueError(f"ramp t_ramp must be positive: {spec}")
            mask_ramp = (t >= t_start) & (t < t_start + t_ramp)
            mask_hold = t >= t_start + t_ramp
            load_matrix[mask_ramp, dof_idx] += amplitude * (t[mask_ramp] - t_start) / t_ramp
            load_matrix[mask_hold, dof_idx] += amplitude

        elif pattern == "gaussian":
            t0 = float(spec.get("t0", 0.1))
            sigma = float(spec.get("sigma", 0.05))
            if sigma <= 0.0:
                raise ValueError(f"gaussian sigma must be positive: {spec}")
            load_matrix[:, dof_idx] += amplitude * np.exp(-0.5 * ((t - t0) / sigma) ** 2)

        else:
            raise ValueError(f"Unsupported load pattern: {pattern}")

    return t, load_matrix
