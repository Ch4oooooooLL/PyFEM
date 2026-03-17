"""
Unit tests for Assembler class.
"""

import pytest
import numpy as np
import scipy.sparse as sp
from PyFEM_Dynamics.solver.assembler import Assembler
from PyFEM_Dynamics.core.element import TrussElement2D
from PyFEM_Dynamics.core.node import Node
from PyFEM_Dynamics.core.material import Material
from PyFEM_Dynamics.core.section import Section


class TestAssembler:
    """Test cases for Assembler class."""
    
    def test_assembler_creation(self, simple_assembly):
        """Test basic assembler creation."""
        assert len(simple_assembly.elements) == 1
        assert simple_assembly.total_dofs == 4
    
    def test_assemble_K_shape(self, simple_assembly):
        """Test global stiffness matrix shape."""
        K = simple_assembly.assemble_K()
        assert K.shape == (4, 4)
    
    def test_assemble_K_sparse(self, simple_assembly):
        """Test that assembled K is sparse matrix."""
        K = simple_assembly.assemble_K()
        assert sp.issparse(K)
        assert isinstance(K, sp.csc_matrix)
    
    def test_assemble_K_symmetric(self, simple_assembly):
        """Test that global K is symmetric."""
        K = simple_assembly.assemble_K()
        K_dense = K.toarray()
        assert np.allclose(K_dense, K_dense.T)
    
    def test_assemble_M_shape(self, simple_assembly):
        """Test global mass matrix shape."""
        M = simple_assembly.assemble_M()
        assert M.shape == (4, 4)
    
    def test_assemble_M_lumped(self, simple_assembly):
        """Test lumped mass matrix assembly."""
        M = simple_assembly.assemble_M(lumping=True)
        M_dense = M.toarray()
        
        # Lumped mass should be diagonal
        assert np.allclose(M_dense, np.diag(np.diag(M_dense)))
    
    def test_two_element_assembly(self, two_element_assembly):
        """Test assembly with two elements."""
        K = two_element_assembly.assemble_K()
        assert K.shape == (6, 6)
        
        # Should be symmetric
        K_dense = K.toarray()
        assert np.allclose(K_dense, K_dense.T)
    
    def test_assembled_values_single_element(self):
        """Test specific values in assembled matrix. single element."""
        material = Material(E=2.1e11, rho=7850)
        section = Section(A=0.01)
        n1, n2 = Node(1, 0, 0), Node(2, 5, 0)
        n1.dofs = [0, 1]
        n2.dofs = [2, 3]
        
        element = TrussElement2D(1, n1, n2, material, section)
        assembler = Assembler([element], total_dofs=4)
        
        K = assembler.assemble_K().toarray()
        
        # Check diagonal entries are positive
        assert K[0, 0] > 0
        assert K[1, 1] >= 0  # Truss has no bending
        assert K[2, 2] > 0
        
        # Check off-diagonal coupling
        assert K[0, 2] < 0  # Negative coupling
        assert K[2, 0] < 0
    
    def test_dofs_assignment_required(self):
        """Test that DOFs must be assigned before assembly."""
        material = Material(E=2.1e11, rho=7850)
        section = Section(A=0.01)
        n1, n2 = Node(1, 0, 0), Node(2, 5, 0)
        # DOFs not assigned!
        
        element = TrussElement2D(1, n1, n2, material, section)
        assembler = Assembler([element], total_dofs=4)
        
        # Should raise error due to empty dofs list
        with pytest.raises(Exception):
            assembler.assemble_K()
