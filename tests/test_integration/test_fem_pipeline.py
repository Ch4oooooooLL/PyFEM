"""
End-to-end integration tests for FEM pipeline.
"""

import pytest
import numpy as np
import tempfile
import os
import sys

from PyFEM_Dynamics.core.node import Node
from PyFEM_Dynamics.core.element import TrussElement2D
from PyFEM_Dynamics.core.material import Material
from PyFEM_Dynamics.core.section import Section
from PyFEM_Dynamics.solver.assembler import Assembler
from PyFEM_Dynamics.solver.boundary import BoundaryCondition
from PyFEM_Dynamics.solver.integrator import StaticSolver


class TestFEMPipeline:
    """Integration tests for complete FEM workflow."""
    
    def test_simple_truss_analysis(self):
        """Test complete analysis of a simple truss structure."""
        # Create nodes
        n1 = Node(1, 0.0, 0.0)
        n2 = Node(2, 5.0, 0.0)
        n3 = Node(3, 2.5, 4.0)
        
        # Assign DOFs (2 DOFs per node for truss)
        n1.dofs = [0, 1]
        n2.dofs = [2, 3]
        n3.dofs = [4, 5]
        
        # Create material and section
        steel = Material(E=2.1e11, rho=7850)
        section = Section(A=0.01)
        
        # Create elements
        elem1 = TrussElement2D(1, n1, n2, steel, section)
        elem2 = TrussElement2D(2, n2, n3, steel, section)
        elem3 = TrussElement2D(3, n3, n1, steel, section)
        
        # Assemble
        assembler = Assembler([elem1, elem2, elem3], total_dofs=6)
        K = assembler.assemble_K()
        
        # Apply boundary conditions (fix nodes 1 and 2)
        bc = BoundaryCondition()
        bc.add_dirichlet_bc(0, 0.0)  # n1.x
        bc.add_dirichlet_bc(1, 0.0)  # n1.y
        bc.add_dirichlet_bc(2, 0.0)  # n2.x
        bc.add_dirichlet_bc(3, 0.0)  # n2.y
        
        # Apply load at node 3
        F = np.zeros(6)
        F[4] = 10000.0  # X force at n3
        F[5] = -5000.0  # Y force at n3
        
        # Solve
        K_mod, F_mod = bc.apply_penalty_method(K, F)
        U = StaticSolver.solve(K_mod, F_mod)
        
        # Verify solution
        assert U.shape == (6,)
        # Fixed nodes should have near-zero displacement
        assert abs(U[0]) < 1e-6
        assert abs(U[1]) < 1e-6
        assert abs(U[2]) < 1e-6
        assert abs(U[3]) < 1e-6
        # Free node should have displacement
        assert abs(U[4]) > 1e-6 or abs(U[5]) > 1e-6
