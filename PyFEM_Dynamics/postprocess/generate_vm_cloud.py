import argparse
import os
import sys

import numpy as np
import yaml
from typing import List, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from pipeline.data_gen import generate_load_matrix, run_fem_solver
from postprocess.plotter import Plotter
from core.io_parser import YAMLParser
from core.node import Node


def _compute_global_plot_limits(
    nodes: List[Node],
    U: np.ndarray,
    scale_factor: float,
    padding_ratio: float = 0.05,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    基于全时程位移，计算统一的坐标显示范围，避免逐帧自动缩放导致视觉失真。
    """
    x_min = float("inf")
    x_max = float("-inf")
    y_min = float("inf")
    y_max = float("-inf")

    for node in nodes:
        ux_hist = U[:, node.dofs[0]]
        uy_hist = U[:, node.dofs[1]]

        x_hist = node.x + ux_hist * scale_factor
        y_hist = node.y + uy_hist * scale_factor

        x_min = min(x_min, float(np.min(x_hist)), float(node.x))
        x_max = max(x_max, float(np.max(x_hist)), float(node.x))
        y_min = min(y_min, float(np.min(y_hist)), float(node.y))
        y_max = max(y_max, float(np.max(y_hist)), float(node.y))

    if not np.isfinite([x_min, x_max, y_min, y_max]).all():
        raise ValueError("Invalid global plotting limits computed from displacement history.")

    x_span = max(x_max - x_min, 1e-12)
    y_span = max(y_max - y_min, 1e-12)
    x_pad = x_span * padding_ratio
    y_pad = y_span * padding_ratio

    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def regenerate_vm_cloud(
    config_path: str,
    output_dir: str,
    sample_id: int = 0,
    frame_step: int = 10,
    representative_frame: int = 100,
    scale_factor: float = 1000.0,
) -> str:
    if frame_step <= 0:
        raise ValueError("frame_step must be a positive integer.")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    structure_file = config.get("structure_file", "structure.yaml")
    if not os.path.isabs(structure_file):
        structure_file = os.path.join(os.path.dirname(config_path), structure_file)

    time_cfg = config["time"]
    dt = float(time_cfg["dt"])
    total_time = float(time_cfg["total_time"])
    num_steps = int(total_time / dt) + 1

    seed = int(config.get("generation", {}).get("random_seed", 42))
    rng = np.random.default_rng(seed + int(sample_id))

    nodes, elements, bcs = YAMLParser.build_structure_objects(structure_file)
    num_nodes = len(nodes)
    dofs_per_node = len(nodes[0].dofs) if nodes else 2

    load_specs = config["load_generation"]["loads"]
    damping_cfg = config.get("damping", {})
    alpha = float(damping_cfg.get("alpha", 0.1))
    beta = float(damping_cfg.get("beta", 0.01))

    load_matrix = generate_load_matrix(load_specs, num_nodes, dofs_per_node, dt, num_steps, rng)
    U, stress_vm = run_fem_solver(nodes, elements, bcs, load_matrix, dt, total_time, alpha, beta)

    sample_dir = os.path.join(output_dir, f"sample_{sample_id:06d}")
    os.makedirs(sample_dir, exist_ok=True)

    vmin = float(np.min(stress_vm))
    vmax = float(np.max(stress_vm))
    xlim, ylim = _compute_global_plot_limits(nodes, U, scale_factor)

    Plotter.setup_paper_style()

    # 1. 按照 step 生成序列帧
    for frame_idx in range(0, stress_vm.shape[0], frame_step):
        frame_path = os.path.join(sample_dir, f"frame_{frame_idx:06d}.png")
        Plotter.plot_truss_vm_frame(
            nodes=nodes,
            elements=elements,
            vm_values=stress_vm[frame_idx],
            title=f"von Mises Stress (sample {sample_id}, step {frame_idx})",
            save_path=frame_path,
            U=U[frame_idx],
            scale_factor=scale_factor,
            vmin=vmin,
            vmax=vmax,
            xlim=xlim,
            ylim=ylim,
        )

    # 2. 确保代表性帧也被生成 (always draw the representative frame)
    # Adjust representative_frame if it's out of bounds
    if representative_frame >= stress_vm.shape[0]:
        representative_frame = stress_vm.shape[0] - 1
    
    rep_path = os.path.join(sample_dir, f"frame_{representative_frame:06d}.png")
    
    Plotter.plot_truss_vm_frame(
        nodes=nodes,
        elements=elements,
        vm_values=stress_vm[representative_frame],
        title=f"von Mises Stress (sample {sample_id}, representative frame {representative_frame})",
        save_path=rep_path,
        U=U[representative_frame],
        scale_factor=scale_factor,
        vmin=vmin,
        vmax=vmax,
        xlim=xlim,
        ylim=ylim,
    )

    return rep_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate dynamic von Mises cloud frames.")
    parser.add_argument("--config", type=str, default=os.path.join(ROOT_DIR, "dataset_config.yaml"))
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(PROJECT_DIR, "postprocess_results", "vm_cloud"),
    )
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--representative-frame", type=int, default=100)
    parser.add_argument("--scale-factor", type=float, default=1000.0)
    args = parser.parse_args()

    rep_path = regenerate_vm_cloud(
        config_path=args.config,
        output_dir=args.output_dir,
        sample_id=args.sample_id,
        frame_step=args.frame_step,
        representative_frame=args.representative_frame,
        scale_factor=args.scale_factor,
    )
    print(f"VM cloud regenerated. Representative frame: {rep_path}")


if __name__ == "__main__":
    main()
