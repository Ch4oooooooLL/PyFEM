import numpy as np
import os
from solver.assembler import Assembler
from solver.boundary import BoundaryCondition
from solver.integrator import StaticSolver
from postprocess.plotter import Plotter
from core.io_parser import IOParser
from core.element import TrussElement2D

def run_single_static_test(material_file=None, structure_file=None, static_load_file=None):
    """
    运行单个静态测试验证，读取结构文件与静载文件。
    """
    if material_file is None:
        material_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "materials.csv")
    if structure_file is None:
        structure_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "structure_input.txt")
    if static_load_file is None:
        static_load_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static_loads.txt")
        
    print("开始运行静力分析测试...")
    
    # 1. 读取基础材料、结构和静载
    materials = IOParser.load_materials(material_file)
    nodes, elements, bcs_local, _ = IOParser.load_structure(structure_file, materials)
    static_loads = IOParser.load_static_loads(static_load_file)
    
    # 初始化
    total_dofs = sum(len(n.dofs) for n in nodes)
    nodes_dict = {n.node_id: n for n in nodes}
    
    # 2. 组装全局矩阵
    assembler = Assembler(elements, total_dofs=total_dofs)
    K = assembler.assemble_K()
    
    # 3. 处理外部静力载荷 F (Fx/Fy)
    F = np.zeros(total_dofs)
    for nid, fx, fy in static_loads:
        if nid not in nodes_dict:
            raise ValueError(f"Static load references undefined node: {nid}")
        if len(nodes_dict[nid].dofs) < 2:
            raise ValueError(f"Node {nid} does not have ux/uy dofs")

        F[nodes_dict[nid].dofs[0]] += fx
        F[nodes_dict[nid].dofs[1]] += fy
    
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
