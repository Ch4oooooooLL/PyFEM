"""
Tests for GTDamagePredictor model.
"""

import pytest
import torch
import numpy as np

# Skip all tests if torch is not available
torch = pytest.importorskip("torch")

from Deep_learning.models.gt_model import GTDamagePredictor


class TestGTDamagePredictor:
    """Test cases for Graph Transformer damage predictor."""
    
    @pytest.fixture
    def model_config(self):
        """Default model configuration."""
        return {
            'num_nodes': 10,
            'num_elements': 15,
            'seq_len': 100,
            'node_in_dim': 2,
            'hidden_dim': 32,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1
        }
    
    @pytest.fixture
    def sample_data(self, model_config):
        """Generate sample input data."""
        batch_size = 4
        num_nodes = model_config['num_nodes']
        seq_len = model_config['seq_len']
        node_in_dim = model_config['node_in_dim']
        num_elements = model_config['num_elements']
        
        # Input: (batch, seq_len, num_nodes * node_in_dim)
        x = torch.randn(batch_size, seq_len, num_nodes * node_in_dim)
        
        # Adjacency matrix: (num_nodes, num_nodes)
        adj = torch.eye(num_nodes) + torch.randn(num_nodes, num_nodes) * 0.1
        adj = (adj > 0).float()
        
        # Edge index: (num_elements, 2)
        edge_index = torch.randint(0, num_nodes, (num_elements, 2))
        
        return x, adj, edge_index
    
    def test_model_creation(self, model_config):
        """Test model can be instantiated."""
        model = GTDamagePredictor(**model_config)
        assert model is not None
        assert model.num_nodes == model_config['num_nodes']
        assert model.num_elements == model_config['num_elements']
    
    def test_forward_pass_shape(self, model_config, sample_data):
        """Test forward pass produces correct output shape."""
        model = GTDamagePredictor(**model_config)
        model.eval()
        
        x, adj, edge_index = sample_data
        
        with torch.no_grad():
            output = model(x, adj, edge_index)
        
        # Output should be (batch_size, num_elements)
        assert output.shape == (4, model_config['num_elements'])
    
    def test_output_range(self, model_config, sample_data):
        """Test output is in valid range [0, 1] (sigmoid)."""
        model = GTDamagePredictor(**model_config)
        model.eval()
        
        x, adj, edge_index = sample_data
        
        with torch.no_grad():
            output = model(x, adj, edge_index)
        
        # After sigmoid, values should be in [0, 1]
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)
    
    def test_model_parameters(self, model_config):
        """Test model has trainable parameters."""
        model = GTDamagePredictor(**model_config)
        params = list(model.parameters())
        
        assert len(params) > 0
        
        # Count total parameters
        total_params = sum(p.numel() for p in params)
        assert total_params > 0
    
    def test_different_batch_sizes(self, model_config):
        """Test model handles different batch sizes."""
        model = GTDamagePredictor(**model_config)
        model.eval()
        
        num_nodes = model_config['num_nodes']
        seq_len = model_config['seq_len']
        node_in_dim = model_config['node_in_dim']
        num_elements = model_config['num_elements']
        
        adj = torch.eye(num_nodes)
        edge_index = torch.randint(0, num_nodes, (num_elements, 2))
        
        for batch_size in [1, 2, 8]:
            x = torch.randn(batch_size, seq_len, num_nodes * node_in_dim)
            
            with torch.no_grad():
                output = model(x, adj, edge_index)
            
            assert output.shape == (batch_size, num_elements)
    
    def test_gradient_flow(self, model_config, sample_data):
        """Test gradients flow through the model."""
        model = GTDamagePredictor(**model_config)
        model.train()
        
        x, adj, edge_index = sample_data
        
        output = model(x, adj, edge_index)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad
