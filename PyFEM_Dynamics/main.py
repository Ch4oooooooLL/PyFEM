import numpy as np
import os
from solver.assembler import Assembler
from solver.boundary import BoundaryCondition
from solver.integrator import StaticSolver
from postprocess.plotter import Plotter
from core.io_parser import IOParser

def run_single_static_test(material_file=None, model_file=None):
    """
    运行单个静态测试验证，自动读取传入模型文件。
    """
    if material_file is None:
        material_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "materials.csv")
    if model_file is None:
        model_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static_input.txt")
        
    print("开始运行静力分析测试...")
    
    # 1. 读取基础材料和模型
    materials = IOParser.load_materials(material_file)
    nodes, elements, bcs_local, loads_local, _, _ = IOParser.load_model(model_file, materials)
    
    # 初始化
    total_dofs = sum(len(n.dofs) for n in nodes)
    nodes_dict = {n.node_id: n for n in nodes}
    
    # 2. 组装全局矩阵
    assembler = Assembler(elements, total_dofs=total_dofs)
    K = assembler.assemble_K()
    
    # 3. 处理外部静力载荷 F
    F = np.zeros(total_dofs)
    for nid, ldof, val in loads_local:
        if nid in nodes_dict and ldof < len(nodes_dict[nid].dofs):
            gdof = nodes_dict[nid].dofs[ldof]
            F[gdof] += val
    
    # 4. 边界处理
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
        
    # 计算每个单元的轴力以用于染色
    axial_forces = np.zeros(len(elements))
    for i, el in enumerate(elements):
        # f = K_local * T * U_element
        u_el = np.array([U[el.node1.dofs[0]], U[el.node1.dofs[1]], U[el.node2.dofs[0]], U[el.node2.dofs[1]]])
        T = el.get_transformation_matrix()
        k_local = el.get_local_stiffness()
        f_local = k_local @ (T @ u_el)
        # 轴向力取 f_local[2] 对应于节点2的轴向力
        axial_forces[i] = f_local[2]
        print(f"单元 {el.element_id} 轴力: {axial_forces[i]:.2f} N")

    # 生成静力变形图
    output_dir = "postprocess_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("\n正在生成结构变形图...")
    Plotter.setup_paper_style()
    Plotter.plot_structure(nodes, elements, U=U, scale_factor=500.0, 
                           element_colors=axial_forces, 
                           title="Truss Static Deformation (Displacement x500)",
                           save_path=os.path.join(output_dir, "static_deformation.png"))
                           
    print("\n测试运行完毕。")


if __name__ == "__main__":
    run_single_static_test()
