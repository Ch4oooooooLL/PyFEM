"""
Pytest configuration and shared fixtures for FEM-DL project.
"""

import pytest
import numpy as np
import sys
import os
import tempfile

try:
    import yaml
except ImportError:
    yaml = None

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import FEM core classes
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'PyFEM_Dynamics'))
from PyFEM_Dynamics.core.node import Node
from PyFEM_Dynamics.core.element import TrussElement2D, BeamElement2D
from PyFEM_Dynamics.core.material import Material
from PyFEM_Dynamics.core.section import Section
from PyFEM_Dynamics.solver.assembler import Assembler
from PyFEM_Dynamics.solver.boundary import BoundaryCondition


# Basic pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


# =============================================================================
# Node Fixtures
# =============================================================================

@pytest.fixture
def simple_node():
    """A simple node at origin."""
    return Node(node_id=1, x=0.0, y=0.0)


@pytest.fixture
def node_at_xy():
    """A node at arbitrary coordinates."""
    return Node(node_id=2, x=3.0, y=4.0)


@pytest.fixture
def two_nodes():
    """Two nodes forming a horizontal line."""
    node1 = Node(node_id=1, x=0.0, y=0.0)
    node2 = Node(node_id=2, x=5.0, y=0.0)
    return node1, node2


@pytest.fixture
def two_nodes_vertical():
    """Two nodes forming a vertical line."""
    node1 = Node(node_id=1, x=0.0, y=0.0)
    node2 = Node(node_id=2, x=0.0, y=3.0)
    return node1, node2


@pytest.fixture
def two_nodes_diagonal():
    """Two nodes forming a diagonal line (3-4-5 triangle)."""
    node1 = Node(node_id=1, x=0.0, y=0.0)
    node2 = Node(node_id=2, x=3.0, y=4.0)
    return node1, node2


# =============================================================================
# Material Fixtures
# =============================================================================

@pytest.fixture
def steel_material():
    """Standard steel material properties."""
    return Material(E=2.1e11, rho=7850.0, nu=0.3)


@pytest.fixture
def aluminum_material():
    """Standard aluminum material properties."""
    return Material(E=7.0e10, rho=2700.0, nu=0.33)


@pytest.fixture
def concrete_material():
    """Standard concrete material properties."""
    return Material(E=3.0e10, rho=2400.0, nu=0.2)


# =============================================================================
# Section Fixtures
# =============================================================================

@pytest.fixture
def square_section():
    """Square cross-section (0.01 m^2 area)."""
    A = 0.01  # 100mm x 100mm
    I = 8.33e-6  # Moment of inertia for square
    return Section(A=A, I=I)


@pytest.fixture
def circular_section():
    """Circular cross-section (radius 0.05m)."""
    r = 0.05
    A = np.pi * r**2
    I = np.pi * r**4 / 4
    return Section(A=A, I=I)


@pytest.fixture
def truss_section():
    """Simple section for truss elements (only area matters)."""
    return Section(A=0.01, I=0.0)


# =============================================================================
# Element Fixtures
# =============================================================================

@pytest.fixture
def horizontal_truss(two_nodes, steel_material, truss_section):
    """A horizontal truss element (length 5.0)."""
    node1, node2 = two_nodes
    return TrussElement2D(
        element_id=1,
        node1=node1,
        node2=node2,
        material=steel_material,
        section=truss_section
    )


@pytest.fixture
def vertical_truss(two_nodes_vertical, steel_material, truss_section):
    """A vertical truss element (length 3.0)."""
    node1, node2 = two_nodes_vertical
    return TrussElement2D(
        element_id=2,
        node1=node1,
        node2=node2,
        material=steel_material,
        section=truss_section
    )


@pytest.fixture
def diagonal_truss(two_nodes_diagonal, aluminum_material, truss_section):
    """A diagonal truss element (length 5.0, 3-4-5 triangle)."""
    node1, node2 = two_nodes_diagonal
    return TrussElement2D(
        element_id=3,
        node1=node1,
        node2=node2,
        material=aluminum_material,
        section=truss_section
    )


@pytest.fixture
def horizontal_beam(two_nodes, steel_material, square_section):
    """A horizontal beam element."""
    node1, node2 = two_nodes
    return BeamElement2D(
        element_id=4,
        node1=node1,
        node2=node2,
        material=steel_material,
        section=square_section
    )


# =============================================================================
# Assembly Fixtures
# =============================================================================

@pytest.fixture
def simple_assembly(horizontal_truss):
    """Simple assembly with one truss element."""
    # Assign DOFs: 2 DOFs per node for truss
    horizontal_truss.node1.dofs = [0, 1]
    horizontal_truss.node2.dofs = [2, 3]
    return Assembler(elements=[horizontal_truss], total_dofs=4)


@pytest.fixture
def two_element_assembly(horizontal_truss, vertical_truss):
    """Assembly with two elements sharing a node."""
    # Share node1 of vertical with node2 of horizontal
    vertical_truss.node1 = horizontal_truss.node2
    
    # Assign DOFs
    horizontal_truss.node1.dofs = [0, 1]
    horizontal_truss.node2.dofs = [2, 3]
    vertical_truss.node2.dofs = [4, 5]
    
    return Assembler(elements=[horizontal_truss, vertical_truss], total_dofs=6)


# =============================================================================
# Boundary Condition Fixtures
# =============================================================================

@pytest.fixture
def fixed_bc():
    """Boundary condition with fixed DOFs."""
    bc = BoundaryCondition()
    bc.add_dirichlet_bc(dof_index=0, value=0.0)
    bc.add_dirichlet_bc(dof_index=1, value=0.0)
    return bc


@pytest.fixture
def roller_bc():
    """Roller support (constrains one DOF)."""
    bc = BoundaryCondition()
    bc.add_dirichlet_bc(dof_index=1, value=0.0)  # Constrain Y movement
    return bc


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Provide a temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_structure_yaml():
    """Sample YAML content for structure configuration."""
    return """
nodes:
  - id: 1
    x: 0.0
    y: 0.0
  - id: 2
    x: 5.0
    y: 0.0
  - id: 3
    x: 2.5
    y: 4.0

materials:
  - id: 1
    E: 2.1e11
    rho: 7850.0
    nu: 0.3

sections:
  - id: 1
    A: 0.01
    I: 8.33e-6

elements:
  - id: 1
    type: truss
    nodes: [1, 2]
    material: 1
    section: 1
  - id: 2
    type: truss
    nodes: [2, 3]
    material: 1
    section: 1
  - id: 3
    type: truss
    nodes: [3, 1]
    material: 1
    section: 1

boundary_conditions:
  - node: 1
    dofs: [0, 1]
    value: 0.0
  - node: 2
    dofs: [1]
    value: 0.0
"""


@pytest.fixture
def mock_yaml_file(temp_dir, sample_structure_yaml):
    """Create a temporary YAML file with structure config."""
    if yaml is None:
        pytest.skip("PyYAML not installed")
    yaml_path = os.path.join(temp_dir, 'structure.yaml')
    with open(yaml_path, 'w') as f:
        f.write(sample_structure_yaml)
    return yaml_path


@pytest.fixture
def sample_load_data():
    """Sample load data for testing."""
    return np.array([[1000.0, 0.0], [0.0, 0.0], [0.0, -500.0]])


@pytest.fixture
def sample_damage_data():
    """Sample damage data for testing."""
    return np.array([0.0, 0.1, 0.05])
