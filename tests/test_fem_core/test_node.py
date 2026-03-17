"""
Unit tests for Node class.
"""

import pytest
import numpy as np
from PyFEM_Dynamics.core.node import Node


class TestNode:
    """Test cases for Node class."""
    
    def test_node_creation(self):
        """Test basic node creation."""
        node = Node(node_id=1, x=1.0, y=2.0)
        assert node.node_id == 1
        assert node.x == 1.0
        assert node.y == 2.0
        assert node.dofs == []
    
    def test_node_at_origin(self, simple_node):
        """Test node at origin coordinates."""
        assert simple_node.node_id == 1
        assert simple_node.x == 0.0
        assert simple_node.y == 0.0
    
    def test_node_dofs_assignment(self):
        """Test DOF assignment to node."""
        node = Node(node_id=1, x=0.0, y=0.0)
        node.dofs = [0, 1]
        assert node.dofs == [0, 1]
    
    def test_node_repr(self):
        """Test node string representation."""
        node = Node(node_id=5, x=3.0, y=4.0)
        repr_str = repr(node)
        assert "Node(id=5" in repr_str
        assert "x=3.0" in repr_str
        assert "y=4.0" in repr_str
    
    def test_multiple_nodes(self):
        """Test creation of multiple nodes."""
        nodes = [
            Node(node_id=i, x=float(i), y=float(i*2))
            for i in range(5)
        ]
        assert len(nodes) == 5
        for i, node in enumerate(nodes):
            assert node.node_id == i
            assert node.x == float(i)
            assert node.y == float(i*2)
    
    def test_negative_coordinates(self):
        """Test node with negative coordinates."""
        node = Node(node_id=1, x=-5.0, y=-3.0)
        assert node.x == -5.0
        assert node.y == -3.0
    
    def test_large_coordinates(self):
        """Test node with large coordinates."""
        node = Node(node_id=1, x=1e6, y=1e-6)
        assert node.x == 1e6
        assert node.y == 1e-6
