"""
Smoke tests for entry points.
"""

import pytest
import sys
import os


class TestEntryPoints:
    """Test that main entry points can be imported and run."""
    
    def test_core_modules_import(self):
        """Test core FEM modules can be imported."""
        from PyFEM_Dynamics.core.node import Node
        from PyFEM_Dynamics.core.element import TrussElement2D
        from PyFEM_Dynamics.core.material import Material
        from PyFEM_Dynamics.core.section import Section
        
        assert Node is not None
        assert TrussElement2D is not None
        assert Material is not None
        assert Section is not None
    
    def test_solver_modules_import(self):
        """Test solver modules can be imported."""
        from PyFEM_Dynamics.solver.assembler import Assembler
        from PyFEM_Dynamics.solver.boundary import BoundaryCondition
        from PyFEM_Dynamics.solver.integrator import StaticSolver
        
        assert Assembler is not None
        assert BoundaryCondition is not None
        assert StaticSolver is not None
    
    def test_pytest_works(self):
        """Test that pytest itself is working."""
        assert True
