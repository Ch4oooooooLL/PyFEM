
import numpy as np
import os
import sys
import yaml

# Add project dir to path
PROJECT_DIR = r"d:\CODE\FEM\PyFEM_Dynamics"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from core.io_parser import YAMLParser
from solver.assembler import Assembler
from solver.boundary import BoundaryCondition
from solver.integrator import StaticSolver

def diagnostic():
    yaml_path = r"d:\CODE\FEM\structure.yaml"
    nodes, elements, bcs = YAMLParser.build_structure_objects(yaml_path)
    
    total_dofs = sum(len(n.dofs) for n in nodes)
    assembler = Assembler(elements, total_dofs)
    K = assembler.assemble_K()
    
    # Static load for comparison
    F = np.zeros(total_dofs)
    # Applied to node 3 (top center)
    # Fx = 10000, Fy = -20000
    node3_dofs = nodes[3].dofs
    F[node3_dofs[0]] = 10000.0
    F[node3_dofs[1]] = -20000.0
    
    bc = BoundaryCondition()
    for nid, ldof, val in bcs:
        gdof = next(n.dofs[ldof] for n in nodes if n.node_id == nid)
        bc.add_dirichlet_bc(gdof, val)
        
    K_mod, F_mod = bc.apply_zero_one_method(K, F)
    U_static = StaticSolver.solve(K_mod, F_mod)
    
    print("Nodes and their assigned DOFs:")
    for n in nodes:
        print(f"Node {n.node_id}: coords=({n.x}, {n.y}), DOFs={n.dofs}")
        
    print("\nStatic Displacements (u, v):")
    for n in nodes:
        u = U_static[n.dofs[0]]
        v = U_static[n.dofs[1]]
        print(f"Node {n.node_id}: ({u:.6e}, {v:.6e})")

    # Dynamic test
    print("\nRunning Dynamic Test (Pulse Load)...")
    from pipeline.data_gen import generate_load_matrix, run_fem_solver
    
    dt = 0.01
    total_time = 2.0
    num_steps = int(total_time / dt) + 1
    rng = np.random.default_rng(42)
    
    # Load spec: Pulse on node 3, Fy = -20000
    load_specs = [{
        'node_id': 3,
        'dof': 'fy',
        'pattern': 'pulse',
        'F0_range': [-20000, -20000],
        't_start_range': [0.0, 0.0],
        't_end_range': [0.5, 0.5]
    }]
    
    load_matrix = generate_load_matrix(load_specs, len(nodes), 2, dt, num_steps, rng)
    U_dyn, _ = run_fem_solver(nodes, elements, bcs, load_matrix, dt, total_time, 0.1, 0.01)
    
    print(f"U_dyn shape: {U_dyn.shape} (Steps x DOFs)")
    max_v3 = np.max(np.abs(U_dyn[:, 7])) # Node 3 Fy is DOF 7
    print(f"Peak vertical displacement at Node 3: {max_v3:.6e} m")
    print(f"Static vertical displacement at Node 3: {abs(U_static[7]):.6e} m")
    print(f"Amplification Ratio: {max_v3 / abs(U_static[7]):.2f}")
    
    print("\nFirst 10 steps vertical displacement at Node 3:")
    print(U_dyn[:10, 7])

if __name__ == "__main__":
    diagnostic()
