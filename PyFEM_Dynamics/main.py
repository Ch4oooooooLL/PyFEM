import numpy as np
import os
from solver.assembler import Assembler
from solver.boundary import BoundaryCondition
from solver.integrator import StaticSolver
from postprocess.plotter import Plotter
from core.io_parser import YAMLParser
from core.element import TrussElement2D

def run_single_static_test(structure_yaml=None):
    """
    运行单个静态测试验证，读取 YAML 结构文件。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    if structure_yaml is None:
        structure_yaml = os.path.join(project_root, "structure.yaml")
        
    print(f"开始运行静力分析测试 (加载: {structure_yaml})...")
    
    # 1. 使用新的 YAML 解析器加载结构
    nodes, elements, bcs_local = YAMLParser.build_structure_objects(structure_yaml)
    
    # 2. 定义静力载荷 (示例：在节点3施加水平10kN，竖向-20kN的力)
    # 在实际应用中，这可以进一步扩展到 YAML 配置中
    static_loads = [(3, 10000.0, -20000.0)] 
    
    # 初始化
    total_dofs = sum(len(n.dofs) for n in nodes)
    nodes_dict = {n.node_id: n for n in nodes}
    
    # 3. 组装全局矩阵
    assembler = Assembler(elements, total_dofs=total_dofs)
    K = assembler.assemble_K()
    
    # 4. 处理外部静力载荷 F (Fx/Fy)
    F = np.zeros(total_dofs)
    for nid, fx, fy in static_loads:
        if nid not in nodes_dict:
            print(f"警告: 载荷引用的节点 {nid} 未定义，跳过。")
            continue
        if len(nodes_dict[nid].dofs) < 2:
            continue

        F[nodes_dict[nid].dofs[0]] += fx
        F[nodes_dict[nid].dofs[1]] += fy
    
    # 5. 边界处理
    bc = BoundaryCondition()
    for nid, ldof, val in bcs_local:
        if nid in nodes_dict and ldof < len(nodes_dict[nid].dofs):
            gdof = nodes_dict[nid].dofs[ldof]
            bc.add_dirichlet_bc(gdof, val)
    
    # 采用划零划一法进行边界约束
    K_mod, F_mod = bc.apply_zero_one_method(K, F)
    
    # 求解
    U = StaticSolver.solve(K_mod, F_mod)
    
    print("\n节点位移计算结果:")
    for i, n in enumerate(nodes):
        print(f"节点 {n.node_id} (u_x, u_y): ({U[n.dofs[0]]:.4e} m, {U[n.dofs[1]]:.4e} m)")
        
    # 计算桁架单元的轴力以用于染色
    axial_forces = np.zeros(len(elements), dtype=float)
    has_only_truss = True
    for i, el in enumerate(elements):
        if isinstance(el, TrussElement2D):
            u_el = np.array([
                U[el.node1.dofs[0]], U[el.node1.dofs[1]],
                U[el.node2.dofs[0]], U[el.node2.dofs[1]]
            ])
            T = el.get_transformation_matrix()
            k_local = el.get_local_stiffness()
            f_local = k_local @ (T @ u_el)
            axial_forces[i] = f_local[2]
            print(f"单元 {el.element_id} 轴力: {axial_forces[i]:.2f} N")
        else:
            has_only_truss = False
            print(f"单元 {el.element_id} 非 Truss2D，跳过轴力后处理。")

    # 生成静力变形图
    output_dir = "postprocess_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("\n正在生成结构变形图...")
    Plotter.setup_paper_style()
    Plotter.plot_structure(nodes, elements, U=U, scale_factor=500.0, 
                           element_colors=axial_forces if has_only_truss else None,
                           title="Truss Static Deformation (Displacement x500)",
                           save_path=os.path.join(output_dir, "static_deformation.png"))
                           
    print("\n测试运行完毕。")


if __name__ == "__main__":
    run_single_static_test()
