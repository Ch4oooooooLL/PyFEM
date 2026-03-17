"""
Unit tests for Section class.
"""

import pytest
import numpy as np
from PyFEM_Dynamics.core.section import Section


class TestSection:
    """Test cases for Section class."""
    
    def test_section_creation(self):
        """Test basic section creation."""
        sec = Section(A=0.01, I=8.33e-6)
        assert sec.A == 0.01
        assert sec.I == 8.33e-6
    
    def test_square_section(self, square_section):
        """Test square section properties."""
        assert square_section.A == 0.01
        assert square_section.I == 8.33e-6
    
    def test_circular_section(self, circular_section):
        """Test circular section properties."""
        r = 0.05
        expected_A = np.pi * r**2
        expected_I = np.pi * r**4 / 4
        
        assert circular_section.A == pytest.approx(expected_A)
        assert circular_section.I == pytest.approx(expected_I)
    
    def test_truss_section(self, truss_section):
        """Test section for truss elements."""
        assert truss_section.A == 0.01
        assert truss_section.I == 0.0
    
    def test_section_repr(self):
        """Test section string representation."""
        sec = Section(A=0.01, I=8.33e-6)
        repr_str = repr(sec)
        assert "Section(" in repr_str
        assert "A=0.01" in repr_str
        assert "I=8.33e-06" in repr_str
    
    def test_zero_section(self):
        """Test section with zero properties."""
        sec = Section(A=0.0, I=0.0)
        assert sec.A == 0.0
        assert sec.I == 0.0
    
    def test_default_inertia(self):
        """Test default moment of inertia (zero)."""
        sec = Section(A=0.01)
        assert sec.A == 0.01
        assert sec.I == 0.0
    
    def test_various_sections(self):
        """Test various cross-section configurations."""
        sections = [
            Section(A=0.001, I=1e-7),   # Small section
            Section(A=0.1, I=1e-3),     # Large section
            Section(A=0.05, I=5e-5),    # Medium section
        ]
        
        for sec in sections:
            assert sec.A > 0
            assert sec.I >= 0
