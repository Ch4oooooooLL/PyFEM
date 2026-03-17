"""
Unit tests for Material class.
"""

import pytest
from PyFEM_Dynamics.core.material import Material


class TestMaterial:
    """Test cases for Material class."""
    
    def test_material_creation(self):
        """Test basic material creation."""
        mat = Material(E=2.1e11, rho=7850.0, nu=0.3)
        assert mat.E == 2.1e11
        assert mat.rho == 7850.0
        assert mat.nu == 0.3
    
    def test_steel_material(self, steel_material):
        """Test steel material properties."""
        assert steel_material.E == 2.1e11
        assert steel_material.rho == 7850.0
        assert steel_material.nu == 0.3
    
    def test_aluminum_material(self, aluminum_material):
        """Test aluminum material properties."""
        assert aluminum_material.E == 7.0e10
        assert aluminum_material.rho == 2700.0
        assert aluminum_material.nu == 0.33
    
    def test_concrete_material(self, concrete_material):
        """Test concrete material properties."""
        assert concrete_material.E == 3.0e10
        assert concrete_material.rho == 2400.0
        assert concrete_material.nu == 0.2
    
    def test_default_poisson_ratio(self):
        """Test default Poisson's ratio value."""
        mat = Material(E=1e9, rho=1000.0)
        assert mat.nu == 0.3
    
    def test_material_repr(self):
        """Test material string representation."""
        mat = Material(E=2e11, rho=7850, nu=0.3)
        repr_str = repr(mat)
        assert "Material(" in repr_str
        assert "E=200000000000.0" in repr_str
        assert "rho=7850" in repr_str
    
    def test_zero_properties(self):
        """Test material with zero properties (edge case)."""
        mat = Material(E=0.0, rho=0.0, nu=0.0)
        assert mat.E == 0.0
        assert mat.rho == 0.0
        assert mat.nu == 0.0
    
    def test_very_large_properties(self):
        """Test material with very large property values."""
        mat = Material(E=1e20, rho=1e10, nu=0.5)
        assert mat.E == 1e20
        assert mat.rho == 1e10
        assert mat.nu == 0.5
