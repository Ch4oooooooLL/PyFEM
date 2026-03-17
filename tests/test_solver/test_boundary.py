"""
Unit tests for BoundaryCondition class.
"""

import pytest
import numpy as np
import scipy.sparse as sp
from PyFEM_Dynamics.solver.boundary import BoundaryCondition


class TestBoundaryCondition:
    """Test cases for BoundaryCondition class."""
    
    def test_bc_creation(self):
        """Test basic BC creation."""
        bc = BoundaryCondition()
        assert bc.dirichlet_bcs == []
    
    def test_add_dirichlet_bc(self):
        """Test adding Dirichlet BC."""
        bc = BoundaryCondition()
        bc.add_dirichlet_bc(dof_index=0, value=0.0)
        assert len(bc.dirichlet_bcs) == 1
        assert bc.dirichlet_bcs[0] == (0, 0.0)
    
    def test_add_multiple_bcs(self):
        """Test adding multiple BCs."""
        bc = BoundaryCondition()
        bc.add_dirichlet_bc(0, 0.0)
        bc.add_dirichlet_bc(1, 0.0)
        bc.add_dirichlet_bc(3, 1.0)  # Non-zero displacement
        
        assert len(bc.dirichlet_bcs) == 3
        assert bc.dirichlet_bcs[2] == (3, 1.0)
    
    def test_fixed_bc_fixture(self, fixed_bc):
        """Test fixed BC fixture."""
        assert len(fixed_bc.dirichlet_bcs) == 2
        assert (0, 0.0) in fixed_bc.dirichlet_bcs
        assert (1, 0.0) in fixed_bc.dirichlet_bcs
    
    def test_roller_bc_fixture(self, roller_bc):
        """Test roller BC fixture."""
        assert len(roller_bc.dirichlet_bcs) == 1
        assert roller_bc.dirichlet_bcs[0] == (1, 0.0)
    
    def test_penalty_method_shape(self):
        """Test penalty method preserves matrix shape."""
        bc = BoundaryCondition()
        bc.add_dirichlet_bc(0, 0.0)
        
        K = sp.csc_matrix(np.array([[1, -1], [-1, 1]], dtype=float))
        F = np.array([0, 1], dtype=float)
        
        K_mod, F_mod = bc.apply_penalty_method(K, F)
        
        assert K_mod.shape == K.shape
        assert F_mod.shape == F.shape
    
    def test_penalty_method_modifies_diagonal(self):
        """Test penalty method modifies diagonal entries."""
        bc = BoundaryCondition()
        bc.add_dirichlet_bc(0, 0.0)
        
        K = sp.csc_matrix(np.array([[1, -1], [-1, 1]], dtype=float))
        F = np.array([0, 1], dtype=float)
        
        K_mod, F_mod = bc.apply_penalty_method(K, F, penalty=1e10)
        K_dense = K_mod.toarray()
        
        # Diagonal should be scaled by penalty
        assert K_dense[0, 0] > 1e9  # Original was 1, now should be ~1e10
    
    def test_zero_one_method_shape(self):
        """Test zero-one method preserves matrix shape."""
        bc = BoundaryCondition()
        bc.add_dirichlet_bc(0, 0.0)
        
        K = sp.csc_matrix(np.array([[2, -1], [-1, 2]], dtype=float))
        F = np.array([1, 1], dtype=float)
        
        K_mod, F_mod = bc.apply_zero_one_method(K, F)
        
        assert K_mod.shape == K.shape
        assert F_mod.shape == F.shape
    
    def test_zero_one_method_sets_diagonal(self):
        """Test zero-one method sets diagonal to 1."""
        bc = BoundaryCondition()
        bc.add_dirichlet_bc(0, 0.0)
        
        K = sp.csc_matrix(np.array([[2, -1], [-1, 2]], dtype=float))
        F = np.array([1, 1], dtype=float)
        
        K_mod, F_mod = bc.apply_zero_one_method(K, F)
        K_dense = K_mod.toarray()
        
        # Constrained DOF should have diagonal = 1
        assert K_dense[0, 0] == 1.0
        assert F_mod[0] == 0.0  # Force set to prescribed value
    
    def test_apply_to_M(self):
        """Test applying BC to mass matrix."""
        bc = BoundaryCondition()
        bc.add_dirichlet_bc(0, 0.0)
        
        M = sp.csc_matrix(np.array([[1, 0], [0, 1]], dtype=float))
        M_mod = bc.apply_to_M(M)
        M_dense = M_mod.toarray()
        
        # Constrained DOF should have diagonal = 1, rest unchanged
        assert M_dense[0, 0] == 1.0
        assert M_dense[1, 1] == 1.0
    
    def test_non_zero_boundary_value(self):
        """Test applying non-zero prescribed displacement."""
        bc = BoundaryCondition()
        bc.add_dirichlet_bc(0, 0.5)  # Prescribed displacement
        
        K = sp.csc_matrix(np.array([[2, -1], [-1, 2]], dtype=float))
        F = np.array([0, 1], dtype=float)
        
        K_mod, F_mod = bc.apply_zero_one_method(K, F)
        
        assert F_mod[0] == 0.5
