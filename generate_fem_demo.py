import os
import sys
import numpy as np
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = SCRIPT_DIR
PYFEM_DIR = os.path.join(ROOT_DIR, "PyFEM_Dynamics")
COND_DIR = os.path.join(ROOT_DIR, "Condition_prediction")
for path in (ROOT_DIR, PYFEM_DIR, COND_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from core.io_parser import YAMLParser
from pipeline.data_gen import run_fem_solver
from Condition_prediction.data.load_builder import build_deterministic_load_matrix


def generate_fem_demo_gif():
    config_path = os.path.join(ROOT_DIR, "dataset_config.yaml")
    structure_file = os.path.join(ROOT_DIR, "structure.yaml")
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    time_cfg = cfg.get("time", {})
    dt = float(time_cfg.get("dt", 0.01))
    total_time = float(time_cfg.get("total_time", 2.0))
    
    damping_cfg = cfg.get("damping", {})
    alpha = float(damping_cfg.get("alpha", 0.1))
    beta = float(damping_cfg.get("beta", 0.01))
    
    nodes, elements, bcs = YAMLParser.build_structure_objects(structure_file)
    num_nodes = len(nodes)
    dofs_per_node = len(nodes[0].dofs) if nodes else 2
    
    np.random.seed(42)
    num_loads = np.random.randint(2, 6)
    candidate_nodes = [i for i in range(num_nodes) if not any(n.dofs == i * dofs_per_node for n in nodes)]
    
    load_specs = []
    for _ in range(num_loads):
        node_id = np.random.choice(candidate_nodes)
        dof = np.random.choice(["fx", "fy"])
        pattern = np.random.choice(["harmonic", "pulse", "half_sine"], p=[0.55, 0.25, 0.20])
        
        F0 = np.random.uniform(5000, 20000)
        freq = np.random.uniform(0.5, 2.4)
        
        spec = {
            "node_id": node_id,
            "dof": dof,
            "pattern": pattern,
            "F0": F0,
            "freq": freq,
            "phase": 0.0,
            "offset": 0.0,
            "duration": total_time
        }
        
        if pattern == "pulse":
            spec["t_start"] = np.random.uniform(0.0, 1.6)
            spec["t_end"] = np.random.uniform(spec["t_start"] + 0.15, 2.0)
        elif pattern == "half_sine":
            spec["t_start"] = np.random.uniform(0.0, 1.6)
            spec["t_end"] = np.random.uniform(spec["t_start"] + 0.15, 2.0)
        elif pattern == "harmonic":
            spec["offset"] = np.random.uniform(-10000, 5000)
        
        load_specs.append(spec)
    
    time_axis, load_matrix = build_deterministic_load_matrix(
        load_specs=load_specs,
        num_nodes=num_nodes,
        dofs_per_node=dofs_per_node,
        dt=dt,
        total_time=total_time,
    )
    
    disp, stress = run_fem_solver(
        nodes,
        elements,
        bcs,
        load_matrix,
        dt,
        total_time,
        damping_alpha=alpha,
        damping_beta=beta,
    )
    
    from Condition_prediction.pipelines.condition_pipeline import _generate_deformation_gif
    
    output_path = os.path.join(ROOT_DIR, "docs", "images", "condition_prediction", "fem_demo_animation.gif")
    _generate_deformation_gif(
        nodes=nodes,
        elements=elements,
        disp_history=disp,
        stress_history=stress,
        time_axis=time_axis,
        save_path=output_path,
        fps=15,
        scale_factor=500.0,
    )
    print(f"Demo GIF saved to: {output_path}")


if __name__ == "__main__":
    generate_fem_demo_gif()
