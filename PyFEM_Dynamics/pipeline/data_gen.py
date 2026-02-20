import os
import numpy as np
import random
from typing import List

from core.element import TrussElement2D
from solver.assembler import Assembler
from solver.boundary import BoundaryCondition
from solver.integrator import NewmarkBetaSolver
from postprocess.plotter import Plotter
from core.io_parser import IOParser
def apply_random_damage(elements: List[TrussElement2D]) -> np.ndarray:
    """
    随机选择 1-3 个单元进行损伤，并返回损伤特征向量 Y
    特征向量为每个单元的 E 值保留系数 (1.0 代表无损，<1 代表受损)
    """
    damage_factors = np.ones(len(elements), dtype=float)
    
    num_damaged = random.randint(1, 3)
    damaged_indices = random.sample(range(len(elements)), num_damaged)
    
    for idx in damaged_indices:
        # 降低弹性模量至 50% 到 90% 之间
        factor = random.uniform(0.5, 0.9)
        damage_factors[idx] = factor
        
        # 实际修改模型参数
        # (需要深拷贝或每次生成新模型，这里通过乘系数修改)
        elements[idx].material.E *= factor
        
    return damage_factors


def generate_dataset(material_file=None, model_file=None):
    """
    数据集生成脚本
    """
    if material_file is None:
        material_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "materials.csv")
    if model_file is None:
        model_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "batch_input.txt")
        
    # 获取基础配置参数
    materials = IOParser.load_materials(material_file)
    base_nodes, _, bcs_local, loads_local, dloads_local, configs = IOParser.load_model(model_file, materials)
    
    num_samples = int(configs.get("num_samples", 100))
    dt = float(configs.get("dt", 0.01))
    total_time = float(configs.get("total_time", 2.0))
    save_dir = configs.get("out_dir", "dataset")
    
    sensor_nodes_str = configs.get("sensor_nodes", "")
    sensor_nids = [int(x.strip()) for x in sensor_nodes_str.split(';')] if sensor_nodes_str else []
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    num_steps = int(total_time / dt) + 1
    total_dofs = sum(len(n.dofs) for n in base_nodes)
    nodes_dict = {n.node_id: n for n in base_nodes}
    
    # 提取传感器所在的全局自由度
    sensor_dofs = []
    for nid in sensor_nids:
        if nid in nodes_dict:
            sensor_dofs.extend(nodes_dict[nid].dofs)
    
    # 构建全局时序载荷矩 global_F
    global_F = np.zeros((total_dofs, num_steps))
    for nid, ldof, val, t_start, t_end in dloads_local:
        if nid in nodes_dict and ldof < len(nodes_dict[nid].dofs):
            gdof = nodes_dict[nid].dofs[ldof]
            start_step = int(t_start / dt)
            end_step = int(t_end / dt)
            global_F[gdof, start_step:min(end_step, num_steps)] = val
            
    # 我们将保存每个样本 X (传感器时间历程) 和 Y (损伤标签)
    X_list = []
    Y_list = []
    
    print(f"正在生成 {num_samples} 个数据样本...")
    
    for i in range(num_samples):
        # 重新解析模型以获得无污染的独立对象
        nodes, elements, bcs_list, _, _, _ = IOParser.load_model(model_file, IOParser.load_materials(material_file))
        
        # 1. 施加损伤以获取标签 Y
        y_label = apply_random_damage(elements)
        
        # 2. 组装全局矩阵
        assembler = Assembler(elements, total_dofs)
        K = assembler.assemble_K()
        M = assembler.assemble_M(lumping=True)
        C = NewmarkBetaSolver.compute_rayleigh_damping(M, K, alpha=0.1, beta=0.01)
        
        # 3. 施加边界条件
        bc = BoundaryCondition()
        curr_nodes_dict = {n.node_id: n for n in nodes}
        for nid, ldof, val in bcs_list:
            if nid in curr_nodes_dict and ldof < len(curr_nodes_dict[nid].dofs):
                gdof = curr_nodes_dict[nid].dofs[ldof]
                bc.add_dirichlet_bc(gdof, val)
                
        # 获取各处理后的矩阵
        K_mod, _ = bc.apply_penalty_method(K, np.zeros(total_dofs))
        M_mod = bc.apply_to_M(M)
        
        # 处理位移约束边界对外力的影响
        F_mod = global_F.copy()
        for nid, ldof, val in bcs_list:
            if nid in curr_nodes_dict and ldof < len(curr_nodes_dict[nid].dofs):
                gdof = curr_nodes_dict[nid].dofs[ldof]
                F_mod[gdof, :] = val
                
        # 4. 求解动力内力
        solver = NewmarkBetaSolver(M_mod, C, K_mod, dt=dt, total_T=total_time)
        U, V, A = solver.solve(F_mod)
        
        # 5. 提取特征数据 X 
        # sensor数据
        if sensor_dofs:
            sensor_data = A[sensor_dofs, :]
        else:
            sensor_data = A
            
        X_list.append(sensor_data)
        Y_list.append(y_label)
        
        if (i + 1) % 50 == 0 or (i + 1) == num_samples:
            print(f"进度: {i + 1}/{num_samples}")
            
    # 将列表全部堆叠成 Numpy 数组
    X_array = np.array(X_list)
    Y_array = np.array(Y_list)
    
    np.save(os.path.join(save_dir, "sensor_X.npy"), X_array)
    np.save(os.path.join(save_dir, "damage_Y.npy"), Y_array)
    
    print(f"数据集生成完成，已保存至 {save_dir}。")
    print(f"特征数据 (X) 维度: {X_array.shape}, 标签数据 (Y) 维度: {Y_array.shape}")
    
    # [后处理与图表生成]: 抽取第一个受损样本进行绘图预览
    print("正在生成第一个样本的可视化图表...")
    sample_index = 0
    time_array = np.linspace(0, total_time, num_steps)
    output_dir = "postprocess_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    Plotter.setup_paper_style()
    Plotter.plot_ai4m_sample(X_array[sample_index], Y_array[sample_index], 
                             sample_idx=sample_index, t_array=time_array,
                             save_path=os.path.join(output_dir, "ai4m_sample_0.png"))

if __name__ == "__main__":
    generate_dataset()
