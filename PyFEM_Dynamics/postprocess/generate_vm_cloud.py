import argparse
import os
import sys

import numpy as np
import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from pipeline.data_gen import generate_load_matrix, run_fem_solver
from postprocess.plotter import Plotter
from core.io_parser import YAMLParser


def regenerate_vm_cloud(
    config_path: str,
    output_dir: str,
    sample_id: int = 0,
    frame_step: int = 10,
    representative_frame: int = 100,
    scale_factor: float = 1000.0,
) -> str:
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
    dofs_per_node = 2

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

    Plotter.setup_paper_style()
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
        )

    return os.path.join(sample_dir, f"frame_{representative_frame:06d}.png")


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
