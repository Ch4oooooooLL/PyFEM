import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from core.element import TrussElement2D
from core.io_parser import YAMLParser
from core.node import Node
from solver.assembler import Assembler
from solver.boundary import BoundaryCondition
from solver.integrator import NewmarkBetaSolver
from solver.stress_recovery import recover_truss_stress_time_history


DOF_TO_LOCAL = {"fx": 0, "ux": 0, "fy": 1, "uy": 1, "rz": 2}


def _sample_scalar(spec: Dict[str, Any], key: str, rng: np.random.Generator, default: float) -> float:
    """
    从配置项中读取标量参数，支持:
    1) key: 直接给定标量
    2) key: [low, high] 范围
    3) key_range: [low, high] 范围
    """
    if key in spec:
        value = spec[key]
    else:
        range_key = f"{key}_range"
        if range_key in spec:
            value = spec[range_key]
        else:
            return float(default)

    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return float(default)
        if len(value) == 1:
            return float(value[0])
        low = float(value[0])
        high = float(value[1])
        return float(rng.uniform(low, high))

    return float(value)


def _normalize_weights(n: int, weights: List[float] | None) -> np.ndarray | None:
    if not weights:
        return None
    arr = np.asarray(weights, dtype=float)
    if arr.size != n:
        return None
    total = float(np.sum(arr))
    if total <= 0.0:
        return None
    return arr / total


def _resolve_candidate_nodes(
    num_nodes: int,
    bcs: List[Tuple[int, int, float]],
    node_cfg: Dict[str, Any],
) -> List[int]:
    mode = str(node_cfg.get("mode", "all_non_support_nodes")).lower()
    include_nodes = [int(n) for n in node_cfg.get("include_nodes", [])]
    exclude_nodes = {int(n) for n in node_cfg.get("exclude_nodes", [])}
    support_nodes = {int(nid) for nid, _, _ in bcs}

    all_nodes = list(range(num_nodes))
    if mode == "all_nodes":
        candidates = all_nodes
    elif mode == "all_non_support_nodes":
        candidates = [n for n in all_nodes if n not in support_nodes]
    elif mode == "explicit":
        candidates = include_nodes
    else:
        raise ValueError(f"Unsupported candidate_nodes.mode: {mode}")

    if include_nodes and mode != "explicit":
        include_set = set(include_nodes)
        candidates = [n for n in candidates if n in include_set]

    candidates = [n for n in candidates if n not in exclude_nodes]
    candidates = sorted(set(candidates))
    if not candidates:
        raise ValueError("No candidate load nodes available after filtering.")
    return candidates


def _build_random_load_specs(
    random_cfg: Dict[str, Any],
    num_nodes: int,
    dofs_per_node: int,
    bcs: List[Tuple[int, int, float]],
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    count_range = random_cfg.get("num_loads_per_sample_range", [1, 3])
    if len(count_range) < 2:
        raise ValueError("num_loads_per_sample_range must contain [min, max].")
    min_loads = int(count_range[0])
    max_loads = int(count_range[1])
    if min_loads <= 0 or max_loads < min_loads:
        raise ValueError("Invalid num_loads_per_sample_range.")

    node_cfg = random_cfg.get("candidate_nodes", {})
    candidate_nodes = _resolve_candidate_nodes(num_nodes, bcs, node_cfg)

    dof_candidates = [str(x).lower() for x in random_cfg.get("dof_candidates", ["fx", "fy"])]
    dof_candidates = [d for d in dof_candidates if d in DOF_TO_LOCAL and DOF_TO_LOCAL[d] < dofs_per_node]
    if not dof_candidates:
        raise ValueError("No valid dof_candidates for current dofs_per_node.")
    dof_prob = _normalize_weights(len(dof_candidates), random_cfg.get("dof_weights"))

    pattern_candidates = [str(x).lower() for x in random_cfg.get("pattern_candidates", ["harmonic"])]
    if not pattern_candidates:
        raise ValueError("pattern_candidates is empty.")
    pattern_prob = _normalize_weights(len(pattern_candidates), random_cfg.get("pattern_weights"))
    param_ranges = random_cfg.get("parameter_ranges", {})
    avoid_duplicate = bool(random_cfg.get("avoid_duplicate_node_dof", True))

    num_loads = int(rng.integers(min_loads, max_loads + 1))
    specs: List[Dict[str, Any]] = []

    if avoid_duplicate:
        all_pairs = [(n, d) for n in candidate_nodes for d in dof_candidates]
        if not all_pairs:
            raise ValueError("No (node, dof) pairs for random loading.")
        num_loads = min(num_loads, len(all_pairs))
        chosen_ids = rng.choice(len(all_pairs), size=num_loads, replace=False)
        pairs = [all_pairs[int(i)] for i in chosen_ids]
    else:
        node_prob = None
        if "node_weights" in random_cfg:
            node_prob = _normalize_weights(len(candidate_nodes), random_cfg.get("node_weights"))
        pairs = []
        for _ in range(num_loads):
            node_id = int(rng.choice(candidate_nodes, p=node_prob))
            dof = str(rng.choice(dof_candidates, p=dof_prob))
            pairs.append((node_id, dof))

    for node_id, dof in pairs:
        pattern = str(rng.choice(pattern_candidates, p=pattern_prob))
        spec: Dict[str, Any] = {"node_id": int(node_id), "dof": dof, "pattern": pattern}
        for key, value in param_ranges.items():
            if isinstance(value, (list, tuple)):
                spec[f"{key}_range"] = list(value)
            else:
                spec[key] = value
        specs.append(spec)

    return specs



def generate_load_matrix(
    load_specs: List[Dict],
    num_nodes: int,
    dofs_per_node: int,
    dt: float,
    num_steps: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    根据载荷配置生成载荷时程矩阵。
    
    返回: shape=(num_steps, num_dofs)
    """
    num_dofs = num_nodes * dofs_per_node
    load_matrix = np.zeros((num_steps, num_dofs), dtype=float)
    t = np.arange(num_steps) * dt
    
    for spec in load_specs:
        node_id = spec['node_id']
        dof = spec['dof']
        pattern = spec['pattern']
        
        dof_idx = node_id * dofs_per_node
        if dof == 'fy' or dof == 'uy':
            dof_idx += 1
        elif dof == 'rz':
            dof_idx += 2
        
        F0 = _sample_scalar(spec, 'F0', rng, default=0.0)
        
        if pattern == 'pulse':
            t_start = _sample_scalar(spec, 't_start', rng, default=0.0)
            t_end = _sample_scalar(spec, 't_end', rng, default=t_start + dt)
            if t_end <= t_start:
                t_end = t_start + 0.1
            
            mask = (t >= t_start) & (t < t_end)
            load_matrix[mask, dof_idx] += F0
            
        elif pattern == 'harmonic':
            freq = _sample_scalar(spec, 'freq', rng, default=1.0)
            phase = _sample_scalar(spec, 'phase', rng, default=0.0)
            offset = _sample_scalar(spec, 'offset', rng, default=0.0)
            duration = float(spec.get('duration', t[-1]))
            duration = min(duration, float(t[-1]))

            signal = F0 * np.sin(2 * np.pi * freq * t + phase) + offset
            mask = t <= duration + 1e-12
            load_matrix[mask, dof_idx] += signal[mask]
            
        elif pattern == 'half_sine':
            t_start = _sample_scalar(spec, 't_start', rng, default=0.0)
            t_end = _sample_scalar(spec, 't_end', rng, default=t_start + dt)
            if t_end <= t_start:
                t_end = t_start + 0.1
            
            duration = t_end - t_start
            mask = (t >= t_start) & (t < t_end)
            t_rel = t[mask] - t_start
            load_matrix[mask, dof_idx] += F0 * np.sin(np.pi * t_rel / duration)
            
        elif pattern == 'ramp':
            t_start = _sample_scalar(spec, 't_start', rng, default=0.0)
            t_ramp = _sample_scalar(spec, 't_ramp', rng, default=0.1)
            mask = (t >= t_start) & (t < t_start + t_ramp)
            load_matrix[mask, dof_idx] += F0 * (t[mask] - t_start) / t_ramp
            mask2 = t >= t_start + t_ramp
            load_matrix[mask2, dof_idx] += F0
            
        elif pattern == 'gaussian':
            t0 = _sample_scalar(spec, 't0', rng, default=0.1)
            sigma = _sample_scalar(spec, 'sigma', rng, default=0.05)
            load_matrix[:, dof_idx] += F0 * np.exp(-0.5 * ((t - t0) / sigma) ** 2)
    
    return load_matrix


def apply_damage(elements: List[TrussElement2D], config: Dict, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    对单元施加随机损伤。
    返回: (damage_vector, E_damaged)
    """
    num_elements = len(elements)
    damage_vec = np.ones(num_elements, dtype=float)
    E_damaged = np.array([el.material.E for el in elements], dtype=float)
    
    if not config.get('enabled', True):
        return damage_vec, E_damaged
    
    min_damaged = config.get('min_damaged_elements', 1)
    max_damaged = config.get('max_damaged_elements', 3)
    reduction_range = config.get('reduction_range', [0.5, 0.9])
    
    num_damaged = int(rng.integers(min_damaged, max_damaged + 1))
    damaged_indices = rng.choice(num_elements, size=num_damaged, replace=False)
    
    for idx in damaged_indices:
        factor = float(rng.uniform(reduction_range[0], reduction_range[1]))
        damage_vec[idx] = factor
        E_damaged[idx] = elements[idx].material.E * factor
        elements[idx].material.E = E_damaged[idx]
    
    return damage_vec, E_damaged


def run_fem_solver(
    nodes: List[Node],
    elements: List[TrussElement2D],
    bcs: List[Tuple[int, int, float]],
    load_matrix: np.ndarray,
    dt: float,
    total_time: float,
    damping_alpha: float,
    damping_beta: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    执行FEM动力学求解。
    返回: (disp, stress) where disp.shape=(num_steps, num_dofs), stress.shape=(num_steps, num_elements)
    """
    total_dofs = sum(len(node.dofs) for node in nodes)
    num_steps = load_matrix.shape[0]
    
    assembler = Assembler(elements, total_dofs)
    K = assembler.assemble_K()
    M = assembler.assemble_M(lumping=True)
    C = NewmarkBetaSolver.compute_rayleigh_damping(M, K, alpha=damping_alpha, beta=damping_beta)
    
    # 5. 边界处理 (改用划零划一法以提高动力学稳定性)
    bc = BoundaryCondition()
    nodes_dict = {node.node_id: node for node in nodes}
    for node_id, local_dof, value in bcs:
        if node_id in nodes_dict and local_dof < len(nodes_dict[node_id].dofs):
            bc.add_dirichlet_bc(nodes_dict[node_id].dofs[local_dof], value)
    
    # 统一处理矩阵，确保边界自由度解耦
    K_mod, _ = bc.apply_zero_one_method(K, np.zeros(total_dofs))
    M_mod = bc.apply_to_M(M)
    
    # 保持阻尼矩阵 C 在边界处的清理逻辑，避免在 K_hat 中引入不必要的耦合项
    C_lil = C.tolil()
    for dof, _ in bc.dirichlet_bcs:
        C_lil.rows[dof] = []
        C_lil.data[dof] = []
    C_mod = C_lil.tocsc()
    
    # 处理荷载向量矩阵
    F_t = load_matrix.T.copy()
    for dof, val in bc.dirichlet_bcs:
        F_t[dof, :] = val  # 通常边界给定值为 0
    
    solver = NewmarkBetaSolver(M_mod, C_mod, K_mod, dt=dt, total_T=total_time)
    U, _, _ = solver.solve(F_t)
    
    sigma_axial, sigma_vm = recover_truss_stress_time_history(elements, U)
    
    return U.T, sigma_vm


def generate_dataset(config_path: str = None) -> None:
    """
    主函数：生成统一格式的训练数据集。
    """
    if config_path is None:
        config_path = os.path.join(ROOT_DIR, 'dataset_config.yaml')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    structure_file = config.get('structure_file', 'structure.yaml')
    if not os.path.isabs(structure_file):
        structure_file = os.path.join(os.path.dirname(config_path), structure_file)
    
    time_config = config['time']
    dt = float(time_config['dt'])
    total_time = float(time_config['total_time'])
    num_steps = int(total_time / dt) + 1
    
    gen_config = config['generation']
    num_samples = int(gen_config['num_samples'])
    random_seed = int(gen_config.get('random_seed', 42))
    rng = np.random.default_rng(random_seed)
    
    output_file = config.get('output_file', 'dataset/train.npz')
    if not os.path.isabs(output_file):
        output_file = os.path.join(os.path.dirname(config_path), output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    nodes, elements, bcs = YAMLParser.build_structure_objects(structure_file)
    num_nodes = len(nodes)
    num_elements = len(elements)
    num_dofs = sum(len(node.dofs) for node in nodes)
    dofs_per_node = len(nodes[0].dofs) if nodes else 2
    
    load_generation_cfg = config['load_generation']
    load_mode = str(load_generation_cfg.get('mode', 'fixed')).lower()
    damage_config = config.get('damage', {})
    damping_config = config.get('damping', {})
    alpha = damping_config.get('alpha', 0.1)
    beta = damping_config.get('beta', 0.01)
    
    all_loads = np.zeros((num_samples, num_steps, num_dofs), dtype=float)
    all_E = np.zeros((num_samples, num_elements), dtype=float)
    all_disp = np.zeros((num_samples, num_steps, num_dofs), dtype=float)
    all_stress = np.zeros((num_samples, num_steps, num_elements), dtype=float)
    all_damage = np.zeros((num_samples, num_elements), dtype=float)
    
    E_base = np.array([el.material.E for el in elements], dtype=float)
    
    print(f"开始生成数据集: {num_samples} 个样本")
    print(f"  结构: {num_nodes} 节点, {num_elements} 单元, {num_dofs} DOF")
    print(f"  时间: {num_steps} 步, dt={dt}s, 总时长={total_time}s")
    
    for i in range(num_samples):
        nodes_i, elements_i, bcs_i = YAMLParser.build_structure_objects(structure_file)
        
        for elem_idx, elem in enumerate(elements_i):
            elem.material.E = E_base[elem_idx]
        
        damage_vec, E_damaged = apply_damage(elements_i, damage_config, rng)
        
        if load_mode in {'random', 'random_multi_point'}:
            random_cfg = load_generation_cfg.get('random_multi_point', {})
            sample_load_specs = _build_random_load_specs(
                random_cfg=random_cfg,
                num_nodes=num_nodes,
                dofs_per_node=dofs_per_node,
                bcs=bcs_i,
                rng=rng,
            )
        elif load_mode in {'fixed', 'manual'}:
            sample_load_specs = load_generation_cfg.get('loads', [])
            if not sample_load_specs:
                raise ValueError("load_generation.loads is empty for fixed/manual mode.")
        else:
            raise ValueError(f"Unsupported load_generation.mode: {load_mode}")

        load_matrix = generate_load_matrix(
            sample_load_specs, num_nodes, dofs_per_node, dt, num_steps, rng
        )
        
        disp, stress = run_fem_solver(
            nodes_i, elements_i, bcs_i, load_matrix, dt, total_time, alpha, beta
        )
        
        all_loads[i] = load_matrix
        all_E[i] = E_damaged
        all_disp[i] = disp
        all_stress[i] = stress
        all_damage[i] = damage_vec
        
        if (i + 1) % 100 == 0 or (i + 1) == num_samples:
            print(f"  进度: {i + 1}/{num_samples}")
    
    np.savez_compressed(
        output_file,
        load=all_loads,
        E=all_E,
        disp=all_disp,
        stress=all_stress,
        damage=all_damage,
    )
    
    metadata = {
        'structure_file': os.path.basename(structure_file),
        'num_samples': num_samples,
        'num_nodes': num_nodes,
        'num_elements': num_elements,
        'num_dofs': num_dofs,
        'dt': dt,
        'total_time': total_time,
        'num_steps': num_steps,
        'random_seed': random_seed,
        'damping_alpha': alpha,
        'damping_beta': beta,
        'load_mode': load_mode,
        'data_keys': ['load', 'E', 'disp', 'stress', 'damage'],
        'shapes': {
            'load': list(all_loads.shape),
            'E': list(all_E.shape),
            'disp': list(all_disp.shape),
            'stress': list(all_stress.shape),
            'damage': list(all_damage.shape),
        }
    }
    
    meta_file = os.path.join(os.path.dirname(output_file), 'metadata.json')
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n数据集生成完成!")
    print(f"  输出文件: {output_file}")
    print(f"  元数据: {meta_file}")
    print(f"\n数据维度:")
    print(f"  load:   {all_loads.shape}  # (N, T, DOF) - 载荷时程")
    print(f"  E:      {all_E.shape}     # (N, elem) - 弹性模量(含损伤)")
    print(f"  disp:   {all_disp.shape}  # (N, T, DOF) - 位移响应")
    print(f"  stress: {all_stress.shape} # (N, T, elem) - 应力响应")
    print(f"  damage: {all_damage.shape} # (N, elem) - 损伤程度(1.0=无损)")


if __name__ == "__main__":
    generate_dataset()
