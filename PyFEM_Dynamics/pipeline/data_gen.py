import json
import os
import sys
from dataclasses import dataclass
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
from core.material import Material
from core.node import Node
from core.section import Section
from solver.assembler import Assembler
from solver.boundary import BoundaryCondition
from solver.integrator import NewmarkBetaSolver
from solver.stress_recovery import recover_truss_stress_time_history


@dataclass
class LoadSpec:
    node_id: int
    dof: str
    pattern: str
    params: Dict[str, Any]





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
        
        F0_range = spec.get('F0_range', [0, 0])
        F0 = float(rng.uniform(F0_range[0], F0_range[1]))
        
        if pattern == 'pulse':
            t_start_range = spec.get('t_start_range', [0, 0])
            t_end_range = spec.get('t_end_range', [0, 0])
            t_start = float(rng.uniform(t_start_range[0], t_start_range[1]))
            t_end = float(rng.uniform(t_end_range[0], t_end_range[1]))
            if t_end <= t_start:
                t_end = t_start + 0.1
            
            mask = (t >= t_start) & (t < t_end)
            load_matrix[mask, dof_idx] += F0
            
        elif pattern == 'harmonic':
            freq = spec.get('freq', 1.0)
            duration = spec.get('duration', t[-1])
            signal = F0 * np.sin(2 * np.pi * freq * t)
            mask = t < duration
            load_matrix[mask, dof_idx] += signal[mask]
            
        elif pattern == 'half_sine':
            t_start_range = spec.get('t_start_range', [0, 0])
            t_end_range = spec.get('t_end_range', [0, 0])
            t_start = float(rng.uniform(t_start_range[0], t_start_range[1]))
            t_end = float(rng.uniform(t_end_range[0], t_end_range[1]))
            if t_end <= t_start:
                t_end = t_start + 0.1
            
            duration = t_end - t_start
            mask = (t >= t_start) & (t < t_end)
            t_rel = t[mask] - t_start
            load_matrix[mask, dof_idx] += F0 * np.sin(np.pi * t_rel / duration)
            
        elif pattern == 'ramp':
            t_start = spec.get('t_start', 0.0)
            t_ramp = spec.get('t_ramp', 0.1)
            mask = (t >= t_start) & (t < t_start + t_ramp)
            load_matrix[mask, dof_idx] += F0 * (t[mask] - t_start) / t_ramp
            mask2 = t >= t_start + t_ramp
            load_matrix[mask2, dof_idx] += F0
            
        elif pattern == 'gaussian':
            t0_range = spec.get('t0_range', [0.1, 0.1])
            sigma_range = spec.get('sigma_range', [0.05, 0.05])
            t0 = float(rng.uniform(t0_range[0], t0_range[1]))
            sigma = float(rng.uniform(sigma_range[0], sigma_range[1]))
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
    
    load_specs = config['load_generation']['loads']
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
        
        load_matrix = generate_load_matrix(
            load_specs, num_nodes, 2, dt, num_steps, rng
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
