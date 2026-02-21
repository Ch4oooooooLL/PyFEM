import csv
import json
import os
import sys
from typing import List, Sequence, Tuple

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from core.element import TrussElement2D
from core.io_parser import DynamicLoadRange, IOParser
from postprocess.plotter import Plotter
from solver.assembler import Assembler
from solver.boundary import BoundaryCondition
from solver.integrator import NewmarkBetaSolver
from solver.stress_recovery import recover_truss_stress_time_history


def apply_random_damage(elements: List[TrussElement2D], rng: np.random.Generator) -> np.ndarray:
    """
    随机选择 1-3 个单元施加刚度折减，并返回损伤标签向量。
    """
    for element in elements:
        if not isinstance(element, TrussElement2D):
            raise NotImplementedError("Dataset generation currently supports TrussElement2D only")

    damage_factors = np.ones(len(elements), dtype=float)
    num_damaged = int(rng.integers(1, 4))
    damaged_indices = rng.choice(len(elements), size=num_damaged, replace=False)

    for idx in damaged_indices:
        factor = float(rng.uniform(0.5, 0.9))
        damage_factors[idx] = factor
        elements[int(idx)].material.E *= factor

    return damage_factors


def _sample_dynamic_loads(
    specs: Sequence[DynamicLoadRange], rng: np.random.Generator, total_time: float
) -> List[Tuple[int, float, float, float, float]]:
    sampled = []
    for spec in specs:
        fx = float(rng.uniform(spec.fx_min, spec.fx_max))
        fy = float(rng.uniform(spec.fy_min, spec.fy_max))

        t_start = None
        t_end = None
        for _ in range(100):
            t_s = float(rng.uniform(spec.t_start_min, spec.t_start_max))
            t_e = float(rng.uniform(spec.t_end_min, spec.t_end_max))
            if t_e > t_s:
                t_start = t_s
                t_end = t_e
                break
        if t_start is None or t_end is None:
            raise ValueError(f"Cannot sample valid t_start/t_end for node {spec.node_id}")

        t_start = max(0.0, min(t_start, total_time))
        t_end = max(0.0, min(t_end, total_time))
        if t_end <= t_start:
            continue

        sampled.append((spec.node_id, fx, fy, t_start, t_end))
    return sampled


def _build_force_matrix(
    nodes,
    sampled_loads: Sequence[Tuple[int, float, float, float, float]],
    dt: float,
    num_steps: int,
) -> np.ndarray:
    total_dofs = sum(len(node.dofs) for node in nodes)
    force_t = np.zeros((total_dofs, num_steps), dtype=float)
    nodes_dict = {node.node_id: node for node in nodes}

    for node_id, fx, fy, t_start, t_end in sampled_loads:
        if node_id not in nodes_dict:
            raise ValueError(f"Dynamic load references undefined node: {node_id}")
        node = nodes_dict[node_id]
        if len(node.dofs) < 2:
            raise ValueError(f"Node {node_id} does not have ux/uy dofs")

        start_step = int(np.floor(t_start / dt))
        end_step = int(np.ceil(t_end / dt))
        start_step = max(0, min(start_step, num_steps - 1))
        end_step = max(0, min(end_step, num_steps))

        if end_step <= start_step:
            continue

        force_t[node.dofs[0], start_step:end_step] += fx
        force_t[node.dofs[1], start_step:end_step] += fy

    return force_t


def _write_displacement_csv(file_path: str, nodes, U: np.ndarray, dt: float):
    with open(file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_step", "time", "node_id", "ux", "uy"])
        num_steps = U.shape[1]
        for step_idx in range(num_steps):
            time_val = step_idx * dt
            for node in nodes:
                writer.writerow(
                    [step_idx, f"{time_val:.12g}", node.node_id, U[node.dofs[0], step_idx], U[node.dofs[1], step_idx]]
                )


def _write_stress_csv(
    file_path: str, elements, sigma_axial: np.ndarray, sigma_vm: np.ndarray, dt: float
):
    with open(file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_step", "time", "element_id", "sigma_axial", "sigma_vm"])
        num_steps = sigma_axial.shape[0]
        for step_idx in range(num_steps):
            time_val = step_idx * dt
            for elem_idx, element in enumerate(elements):
                writer.writerow(
                    [
                        step_idx,
                        f"{time_val:.12g}",
                        element.element_id,
                        sigma_axial[step_idx, elem_idx],
                        sigma_vm[step_idx, elem_idx],
                    ]
                )


def _parse_int_set(value: str) -> set:
    if not value:
        return set()
    return {int(x.strip()) for x in value.split(";") if x.strip()}


def generate_dataset(material_file=None, structure_file=None, dynamic_load_file=None):
    """
    批量生成位移训练集，并输出应力历史与 VM 云图。
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if material_file is None:
        material_file = os.path.join(root_dir, "materials.csv")
    if structure_file is None:
        structure_file = os.path.join(root_dir, "structure_input.txt")
    if dynamic_load_file is None:
        dynamic_load_file = os.path.join(root_dir, "dynamic_loads.txt")

    dynamic_specs, configs = IOParser.load_dynamic_load_specs(dynamic_load_file)
    if not dynamic_specs:
        raise ValueError("dynamic_loads file does not contain any DLOAD_RANGE records")

    num_samples = int(configs.get("num_samples", 100))
    dt = float(configs.get("dt", 0.01))
    total_time = float(configs.get("total_time", 2.0))
    save_dir = configs.get("out_dir", "dataset")
    random_seed = int(configs.get("random_seed", 42))
    cloud_sample_ids = _parse_int_set(configs.get("cloud_sample_ids", "0"))
    cloud_step_stride = max(1, int(configs.get("cloud_step_stride", 1)))
    cloud_out_dir = configs.get("cloud_out_dir", os.path.join("postprocess_results", "vm_cloud"))
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(root_dir, save_dir)
    if not os.path.isabs(cloud_out_dir):
        cloud_out_dir = os.path.join(root_dir, cloud_out_dir)

    num_steps = int(total_time / dt) + 1
    rng = np.random.default_rng(random_seed)

    displacement_dir = os.path.join(save_dir, "displacement")
    stress_dir = os.path.join(save_dir, "stress")
    labels_dir = os.path.join(save_dir, "labels")
    meta_dir = os.path.join(save_dir, "meta")
    os.makedirs(displacement_dir, exist_ok=True)
    os.makedirs(stress_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(cloud_out_dir, exist_ok=True)

    # 用于输出 meta
    materials = IOParser.load_materials(material_file)
    base_nodes, base_elements, _, _ = IOParser.load_structure(structure_file, materials)
    if not all(isinstance(el, TrussElement2D) for el in base_elements):
        raise NotImplementedError("Batch stress recovery supports Truss2D-only models")

    print(f"正在生成 {num_samples} 个数据样本...")
    damage_labels = []
    Plotter.setup_paper_style()

    for sample_idx in range(num_samples):
        materials_i = IOParser.load_materials(material_file)
        nodes, elements, bcs_list, _ = IOParser.load_structure(structure_file, materials_i)
        if not all(isinstance(el, TrussElement2D) for el in elements):
            raise NotImplementedError("Batch dataset generation supports Truss2D-only models")

        total_dofs = sum(len(node.dofs) for node in nodes)
        y_label = apply_random_damage(elements, rng)

        assembler = Assembler(elements, total_dofs)
        K = assembler.assemble_K()
        M = assembler.assemble_M(lumping=True)
        C = NewmarkBetaSolver.compute_rayleigh_damping(M, K, alpha=0.1, beta=0.01)

        sampled_loads = _sample_dynamic_loads(dynamic_specs, rng, total_time)
        F_t = _build_force_matrix(nodes, sampled_loads, dt, num_steps)

        bc = BoundaryCondition()
        nodes_dict = {node.node_id: node for node in nodes}
        for node_id, local_dof, value in bcs_list:
            if node_id in nodes_dict and local_dof < len(nodes_dict[node_id].dofs):
                bc.add_dirichlet_bc(nodes_dict[node_id].dofs[local_dof], value)

        K_mod, _ = bc.apply_penalty_method(K, np.zeros(total_dofs))
        M_mod = bc.apply_to_M(M)

        F_mod = F_t.copy()
        for node_id, local_dof, value in bcs_list:
            if node_id in nodes_dict and local_dof < len(nodes_dict[node_id].dofs):
                F_mod[nodes_dict[node_id].dofs[local_dof], :] = value

        solver = NewmarkBetaSolver(M_mod, C, K_mod, dt=dt, total_T=total_time)
        U, _, _ = solver.solve(F_mod)

        disp_csv_path = os.path.join(displacement_dir, f"sample_{sample_idx:06d}.csv")
        _write_displacement_csv(disp_csv_path, nodes, U, dt)

        sigma_axial, sigma_vm = recover_truss_stress_time_history(elements, U)
        stress_csv_path = os.path.join(stress_dir, f"sample_{sample_idx:06d}.csv")
        _write_stress_csv(stress_csv_path, elements, sigma_axial, sigma_vm, dt)

        if sample_idx in cloud_sample_ids:
            sample_cloud_dir = os.path.join(cloud_out_dir, f"sample_{sample_idx:06d}")
            os.makedirs(sample_cloud_dir, exist_ok=True)
            vm_min = float(np.min(sigma_vm))
            vm_max = float(np.max(sigma_vm))
            for step_idx in range(0, num_steps, cloud_step_stride):
                title = f"Sample {sample_idx} - t={step_idx * dt:.4f}s"
                save_path = os.path.join(sample_cloud_dir, f"frame_{step_idx:06d}.png")
                Plotter.plot_truss_vm_frame(
                    nodes,
                    elements,
                    sigma_vm[step_idx, :],
                    title=title,
                    save_path=save_path,
                    vmin=vm_min,
                    vmax=vm_max,
                )

        damage_labels.append(y_label)

        if (sample_idx + 1) % 10 == 0 or (sample_idx + 1) == num_samples:
            print(f"进度: {sample_idx + 1}/{num_samples}")

    y_array = np.array(damage_labels, dtype=float)
    np.save(os.path.join(labels_dir, "damage_Y.npy"), y_array)

    meta = {
        "num_samples": num_samples,
        "dt": dt,
        "total_time": total_time,
        "num_steps": num_steps,
        "random_seed": random_seed,
        "num_nodes": len(base_nodes),
        "num_elements": len(base_elements),
        "material_file": material_file,
        "structure_file": structure_file,
        "dynamic_load_file": dynamic_load_file,
        "cloud_sample_ids": sorted(list(cloud_sample_ids)),
        "cloud_step_stride": cloud_step_stride,
        "displacement_dir": displacement_dir,
        "stress_dir": stress_dir,
        "labels_dir": labels_dir,
        "cloud_out_dir": cloud_out_dir,
    }
    with open(os.path.join(meta_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"数据集生成完成，已保存至 {save_dir}。")
    print(f"损伤标签 (Y) 维度: {y_array.shape}")

if __name__ == "__main__":
    generate_dataset()
