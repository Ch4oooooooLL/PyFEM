"""
Unit tests for integrator classes.
"""

import pytest
import numpy as np
import scipy.sparse as sp
from PyFEM_Dynamics.solver.integrator import StaticSolver, NewmarkBetaSolver


class TestStaticSolver:
    """Test cases for StaticSolver."""
    
    def test_static_solve_basic(self):
        """Test basic static solution."""
        # Simple 2x2 system: [2 -1; -1 2] * u = [1; 0]
        K = sp.csc_matrix(np.array([[2, -1], [-1, 2]], dtype=float))
        F = np.array([1.0, 0.0])
        
        U = StaticSolver.solve(K, F)
        
        assert U.shape == (2,)
        # Verify solution: K*U = F
        residual = K.dot(U) - F
        assert np.allclose(residual, 0)
    
    def test_static_solve_larger_system(self):
        """Test static solve with larger system."""
        # 3x3 tridiagonal system
        K = sp.csc_matrix(np.array([
            [2, -1, 0],
            [-1, 2, -1],
            [0, -1, 2]
        ], dtype=float))
        F = np.array([1.0, 0.0, 0.0])
        
        U = StaticSolver.solve(K, F)
        
        assert U.shape == (3,)
        residual = K.dot(U) - F
        assert np.allclose(residual, 0)


class TestNewmarkBetaSolver:
    """Test cases for NewmarkBetaSolver."""
    
    def test_solver_creation(self):
        """Test basic solver creation."""
        M = sp.eye(2, format='csc')
        C = sp.csc_matrix((2, 2))
        K = sp.eye(2, format='csc')
        
        solver = NewmarkBetaSolver(M, C, K, dt=0.01, total_T=1.0)
        
        assert solver.dt == 0.01
        assert solver.num_steps == 101  # int(1.0/0.01) + 1
        assert solver.gamma == 0.5
        assert solver.beta == 0.25
    
    def test_newmark_constants_average_acceleration(self):
        """Test Newmark constants for average acceleration method."""
        M = sp.eye(2, format='csc')
        C = sp.csc_matrix((2, 2))
        K = sp.eye(2, format='csc')
        
        dt = 0.01
        solver = NewmarkBetaSolver(M, C, K, dt=dt, total_T=0.1, gamma=0.5, beta=0.25)
        
        # Check constants calculation
        assert solver.dt == dt
        # gamma = 0.5, beta = 0.25 for average acceleration
    
    def test_solve_free_vibration(self):
        """Test free vibration solution (no external force)."""
        # Simple 1-DOF system
        M = sp.eye(1, format='csc')
        C = sp.csc_matrix((1, 1))
        K = sp.eye(1, format='csc')
        
        solver = NewmarkBetaSolver(M, C, K, dt=0.01, total_T=0.1)
        
        # Zero force
        F_t = np.zeros((1, solver.num_steps))
        
        # Initial displacement
        U0 = np.array([1.0])
        V0 = np.array([0.0])
        
        U, V, A = solver.solve(F_t, U0, V0)
        
        assert U.shape == (1, solver.num_steps)
        assert V.shape == (1, solver.num_steps)
        assert A.shape == (1, solver.num_steps)
        
        # Check initial conditions
        assert np.isclose(U[0, 0], 1.0)
        assert np.isclose(V[0, 0], 0.0)
    
    def test_solve_with_constant_force(self):
        """Test solution with constant force."""
        M = sp.eye(2, format='csc')
        C = sp.csc_matrix((2, 2))
        K = sp.eye(2, format='csc')
        
        solver = NewmarkBetaSolver(M, C, K, dt=0.01, total_T=0.1)
        
        # Constant force
        F_t = np.ones((2, solver.num_steps))
        
        U, V, A = solver.solve(F_t)
        
        assert U.shape == (2, solver.num_steps)
        # System should start moving from zero
        assert not np.allclose(U[:, 0], U[:, -1])
    
    def test_rayleigh_damping(self):
        """Test Rayleigh damping matrix computation."""
        M = sp.eye(2, format='csc')
        K = sp.csc_matrix(np.array([[2, -1], [-1, 2]], dtype=float))
        
        alpha = 0.1
        beta = 0.01
        
        C = NewmarkBetaSolver.compute_rayleigh_damping(M, K, alpha, beta)
        
        # C = alpha*M + beta*K
        expected_C = alpha * M + beta * K
        assert np.allclose(C.toarray(), expected_C.toarray())
    
    def test_time_stepping_consistency(self):
        """Test that time stepping produces consistent results."""
        M = sp.eye(2, format='csc')
        C = sp.csc_matrix((2, 2))
        K = sp.eye(2, format='csc')
        
        solver = NewmarkBetaSolver(M, C, K, dt=0.01, total_T=0.05)
        F_t = np.zeros((2, solver.num_steps))
        
        U, V, A = solver.solve(F_t)
        
        # All arrays should have correct shape
        assert U.shape == (2, 6)  # 0.05/0.01 + 1 = 6 steps
        assert V.shape == (2, 6)
        assert A.shape == (2, 6)
