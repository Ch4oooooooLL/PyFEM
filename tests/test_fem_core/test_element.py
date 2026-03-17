"""
Unit tests for Element classes.
"""

import pytest
import numpy as np
from PyFEM_Dynamics.core.element import TrussElement2D, BeamElement2D
from PyFEM_Dynamics.core.node import Node
from PyFEM_Dynamics.core.material import Material
from PyFEM_Dynamics.core.section import Section


class TestTrussElement:
    """Test cases for TrussElement2D."""
    
    def test_truss_element_creation(self, horizontal_truss):
        """Test basic truss element creation."""
        assert horizontal_truss.element_id == 1
        assert horizontal_truss.length == 5.0
        assert horizontal_truss.angle == 0.0
    
    def test_truss_length_calculation(self, diagonal_truss):
        """Test length calculation for diagonal truss."""
        # 3-4-5 triangle
        assert diagonal_truss.length == pytest.approx(5.0)
    
    def test_truss_angle_calculation(self):
        """Test angle calculation for different orientations."""
        material = Material(E=2e11, rho=7850)
        section = Section(A=0.01)
        
        # Horizontal
        n1, n2 = Node(1, 0, 0), Node(2, 5, 0)
        truss = TrussElement2D(1, n1, n2, material, section)
        assert truss.angle == 0.0
        
        # Vertical
        n3, n4 = Node(3, 0, 0), Node(4, 0, 3)
        truss_v = TrussElement2D(2, n3, n4, material, section)
        assert truss_v.angle == pytest.approx(np.pi/2)
        
        # 45 degrees
        n5, n6 = Node(5, 0, 0), Node(6, 1, 1)
        truss_45 = TrussElement2D(3, n5, n6, material, section)
        assert truss_45.angle == pytest.approx(np.pi/4)
    
    def test_local_stiffness_shape(self, horizontal_truss):
        """Test local stiffness matrix shape."""
        k_local = horizontal_truss.get_local_stiffness()
        assert k_local.shape == (4, 4)
    
    def test_local_stiffness_values(self, horizontal_truss):
        """Test local stiffness matrix values."""
        k_local = horizontal_truss.get_local_stiffness()
        E = 2.1e11
        A = 0.01
        L = 5.0
        k_expected = E * A / L
        
        assert k_local[0, 0] == pytest.approx(k_expected)
        assert k_local[0, 2] == pytest.approx(-k_expected)
        assert k_local[2, 0] == pytest.approx(-k_expected)
        assert k_local[2, 2] == pytest.approx(k_expected)
    
    def test_transformation_matrix_shape(self, horizontal_truss):
        """Test transformation matrix shape."""
        T = horizontal_truss.get_transformation_matrix()
        assert T.shape == (4, 4)
    
    def test_transformation_matrix_orthogonal(self, diagonal_truss):
        """Test that transformation matrix is orthogonal."""
        T = diagonal_truss.get_transformation_matrix()
        # T^T * T should be identity
        identity = T.T @ T
        assert np.allclose(identity, np.eye(4))
    
    def test_global_stiffness_shape(self, horizontal_truss):
        """Test global stiffness matrix shape."""
        k_global = horizontal_truss.get_global_stiffness()
        assert k_global.shape == (4, 4)
    
    def test_global_stiffness_symmetric(self, horizontal_truss):
        """Test that global stiffness matrix is symmetric."""
        k_global = horizontal_truss.get_global_stiffness()
        assert np.allclose(k_global, k_global.T)
    
    def test_local_mass_shape(self, horizontal_truss):
        """Test local mass matrix shape."""
        m_local = horizontal_truss.get_local_mass()
        assert m_local.shape == (4, 4)
    
    def test_local_mass_lumped(self, horizontal_truss):
        """Test lumped mass matrix."""
        m_lumped = horizontal_truss.get_local_mass(lumping=True)
        rho = 7850.0
        A = 0.01
        L = 5.0
        m_expected = rho * A * L / 2.0
        
        assert np.allclose(m_lumped, np.diag([m_expected, m_expected, m_expected, m_expected]))
    
    def test_global_mass_shape(self, horizontal_truss):
        """Test global mass matrix shape."""
        m_global = horizontal_truss.get_global_mass()
        assert m_global.shape == (4, 4)


class TestBeamElement:
    """Test cases for BeamElement2D."""
    
    def test_beam_element_creation(self, horizontal_beam):
        """Test basic beam element creation."""
        assert horizontal_beam.element_id == 4
        assert horizontal_beam.length == 5.0
    
    def test_local_stiffness_shape(self, horizontal_beam):
        """Test local stiffness matrix shape for beam."""
        k_local = horizontal_beam.get_local_stiffness()
        assert k_local.shape == (6, 6)
    
    def test_local_stiffness_symmetric(self, horizontal_beam):
        """Test that beam stiffness matrix is symmetric."""
        k_local = horizontal_beam.get_local_stiffness()
        assert np.allclose(k_local, k_local.T)
    
    def test_transformation_matrix_shape(self, horizontal_beam):
        """Test transformation matrix shape for beam."""
        T = horizontal_beam.get_transformation_matrix()
        assert T.shape == (6, 6)
    
    def test_global_stiffness_shape(self, horizontal_beam):
        """Test global stiffness matrix shape for beam."""
        k_global = horizontal_beam.get_global_stiffness()
        assert k_global.shape == (6, 6)
    
    def test_local_mass_shape(self, horizontal_beam):
        """Test local mass matrix shape for beam."""
        m_local = horizontal_beam.get_local_mass()
        assert m_local.shape == (6, 6)
    
    def test_beam_vs_truss_dofs(self):
        """Test that beam has different DOFs than truss."""
        material = Material(E=2e11, rho=7850)
        section = Section(A=0.01, I=8.33e-6)
        n1, n2 = Node(1, 0, 0), Node(2, 5, 0)
        
        truss = TrussElement2D(1, n1, n2, material, Section(A=0.01))
        beam = BeamElement2D(2, n1, n2, material, section)
        
        # Truss: 2 DOFs per node = 4x4 matrices
        # Beam: 3 DOFs per node = 6x6 matrices
        assert truss.get_local_stiffness().shape == (4, 4)
        assert beam.get_local_stiffness().shape == (6, 6)
