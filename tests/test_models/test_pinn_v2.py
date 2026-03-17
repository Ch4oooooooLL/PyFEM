"""
Unit tests for PINN V2 model.
"""

import pytest
import torch
import numpy as np

from Deep_learning.models.pinn_model_v2 import (
    ResidualBlock, PINNDamagePredictorV2, PINNEnsemble
)


class TestResidualBlock:
    """Test cases for ResidualBlock."""
    
    def test_initialization(self):
        """Test residual block initialization."""
        block = ResidualBlock(dim=64, dropout=0.3)
        assert isinstance(block.net, torch.nn.Sequential)
    
    def test_forward(self):
        """Test forward pass preserves shape."""
        block = ResidualBlock(dim=64)
        x = torch.randn(4, 64)
        out = block(x)
        
        assert out.shape == x.shape
        assert not torch.allclose(out, x)  # Should be modified by residual
    
    def test_skip_connection(self):
        """Test that skip connection works."""
        block = ResidualBlock(dim=32)
        x = torch.randn(2, 32)
        
        # Check that output is x + f(x)
        residual = block.net(x)
        expected = x + residual
        actual = block(x)
        
        assert torch.allclose(actual, expected, atol=1e-6)


class TestPINNDamagePredictorV2:
    """Test cases for PINNDamagePredictorV2."""
    
    def test_initialization_defaults(self):
        """Test model initialization with defaults."""
        model = PINNDamagePredictorV2(
            input_dim=20,
            output_dim=11
        )
        
        assert model.input_dim == 20
        assert model.output_dim == 11
        assert model.use_physics is True
    
    def test_initialization_custom(self):
        """Test model initialization with custom parameters."""
        model = PINNDamagePredictorV2(
            input_dim=20,
            hidden_dims=[128, 128],
            output_dim=11,
            num_residual_blocks=3,
            dropout=0.2,
            use_physics=False
        )
        
        assert model.use_physics is False
        assert model.physics_weight == 0.1
    
    def test_forward_2d_input(self):
        """Test forward pass with 2D input."""
        model = PINNDamagePredictorV2(
            input_dim=20,
            output_dim=11
        )
        model.eval()
        
        x = torch.randn(4, 20)
        damage = model(x)
        
        assert damage.shape == (4, 11)
        assert torch.all(damage >= 0) and torch.all(damage <= 1)  # Sigmoid output
    
    def test_forward_3d_input(self):
        """Test forward pass with 3D input (time dimension)."""
        model = PINNDamagePredictorV2(
            input_dim=20,
            output_dim=11
        )
        model.eval()
        
        x = torch.randn(4, 10, 20)  # (batch, time, features)
        damage = model(x)
        
        assert damage.shape == (4, 11)
    
    def test_output_range(self):
        """Test that output is in valid range [0, 1]."""
        model = PINNDamagePredictorV2(input_dim=20, output_dim=11)
        model.eval()
        
        # Test with extreme inputs
        x = torch.randn(100, 20) * 10
        damage = model(x)
        
        assert torch.all(damage >= 0)
        assert torch.all(damage <= 1)
    
    def test_set_structure_info(self):
        """Test setting structure information."""
        model = PINNDamagePredictorV2(input_dim=20, output_dim=11)
        
        adjacency = np.random.randint(0, 2, size=(11, 11)).astype(np.float32)
        stiffness = np.random.rand(11).astype(np.float32)
        coords = np.random.rand(11, 2).astype(np.float32)
        
        model.set_structure_info(adjacency, stiffness, coords)
        
        assert hasattr(model, 'element_adjacency')
        assert hasattr(model, 'stiffness_importance')
        assert hasattr(model, 'element_coords')
    
    def test_physics_loss_without_structure(self):
        """Test physics loss computation without structure info."""
        model = PINNDamagePredictorV2(input_dim=20, output_dim=11)
        model.eval()
        
        damage_pred = torch.rand(4, 11)
        losses = model.physics_loss(damage_pred)
        
        assert 'total_physics_loss' in losses
        assert 'range_loss' in losses
        assert 'sparsity_loss' in losses
    
    def test_physics_loss_with_structure(self):
        """Test physics loss computation with structure info."""
        model = PINNDamagePredictorV2(input_dim=20, output_dim=11)
        
        adjacency = np.eye(11, dtype=np.float32)
        adjacency[0, 1] = adjacency[1, 0] = 1
        model.set_structure_info(adjacency)
        
        damage_pred = torch.rand(4, 11)
        losses = model.physics_loss(damage_pred)
        
        assert 'smoothness_loss' in losses
        assert 'total_physics_loss' in losses
    
    def test_physics_loss_disabled(self):
        """Test physics loss when physics is disabled."""
        model = PINNDamagePredictorV2(
            input_dim=20,
            output_dim=11,
            use_physics=False
        )
        
        damage_pred = torch.rand(4, 11)
        losses = model.physics_loss(damage_pred)
        
        assert losses['total_physics_loss'].item() == 0.0
    
    def test_compute_total_loss(self):
        """Test total loss computation."""
        model = PINNDamagePredictorV2(input_dim=20, output_dim=11)
        model.eval()
        
        damage_pred = torch.rand(4, 11)
        damage_true = torch.rand(4, 11)
        
        total_loss, loss_dict = model.compute_total_loss(
            damage_pred, damage_true,
            data_weight=1.0,
            physics_weight=0.1
        )
        
        assert isinstance(total_loss, torch.Tensor)
        assert 'total_loss' in loss_dict
        assert 'data_loss' in loss_dict
        assert 'physics_loss' in loss_dict
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = PINNDamagePredictorV2(input_dim=20, output_dim=11)
        
        x = torch.randn(4, 20, requires_grad=True)
        damage = model(x)
        loss = damage.sum()
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        
        # Check that model parameters have gradients
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad
    
    def test_different_batch_sizes(self):
        """Test model with different batch sizes."""
        model = PINNDamagePredictorV2(input_dim=20, output_dim=11)
        model.eval()
        
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 20)
            damage = model(x)
            assert damage.shape == (batch_size, 11)


class TestPINNEnsemble:
    """Test cases for PINNEnsemble."""
    
    def test_initialization(self):
        """Test ensemble initialization."""
        ensemble = PINNEnsemble(
            input_dim=20,
            output_dim=11,
            num_models=3
        )
        
        assert len(ensemble.models) == 3
        assert all(isinstance(m, PINNDamagePredictorV2) for m in ensemble.models)
    
    def test_forward(self):
        """Test ensemble forward pass."""
        ensemble = PINNEnsemble(input_dim=20, output_dim=11, num_models=3)
        ensemble.eval()
        
        x = torch.randn(4, 20)
        damage = ensemble(x)
        
        assert damage.shape == (4, 11)
        assert torch.all(damage >= 0) and torch.all(damage <= 1)
    
    def test_predict_with_uncertainty(self):
        """Test prediction with uncertainty."""
        ensemble = PINNEnsemble(input_dim=20, output_dim=11, num_models=3)
        ensemble.eval()
        
        x = torch.randn(4, 20)
        mean, uncertainty = ensemble.predict_with_uncertainty(x)
        
        assert mean.shape == (4, 11)
        assert uncertainty.shape == (4, 11)
        assert torch.all(uncertainty >= 0)
    
    def test_ensemble_averaging(self):
        """Test that ensemble properly averages predictions."""
        ensemble = PINNEnsemble(input_dim=20, output_dim=11, num_models=2)
        ensemble.eval()
        
        # Get individual predictions
        x = torch.randn(4, 20)
        preds = [m(x) for m in ensemble.models]
        expected_mean = torch.stack(preds).mean(dim=0)
        
        # Get ensemble prediction
        actual_mean = ensemble(x)
        
        assert torch.allclose(actual_mean, expected_mean, atol=1e-6)


class TestModelSaveLoad:
    """Test model saving and loading."""
    
    def test_save_load_state_dict(self):
        """Test saving and loading model state."""
        model1 = PINNDamagePredictorV2(input_dim=20, output_dim=11)
        model1.eval()
        
        # Get prediction before saving
        x = torch.randn(4, 20)
        pred1 = model1(x)
        
        # Save and load
        state_dict = model1.state_dict()
        model2 = PINNDamagePredictorV2(input_dim=20, output_dim=11)
        model2.load_state_dict(state_dict)
        model2.eval()
        
        # Check predictions match
        pred2 = model2(x)
        assert torch.allclose(pred1, pred2, atol=1e-6)
    
    def test_save_load_with_structure_info(self):
        """Test saving and loading with structure info."""
        model1 = PINNDamagePredictorV2(input_dim=20, output_dim=11)
        
        adjacency = np.eye(11, dtype=np.float32)
        model1.set_structure_info(adjacency)
        
        # Save and load
        state_dict = model1.state_dict()
        model2 = PINNDamagePredictorV2(input_dim=20, output_dim=11)
        model2.set_structure_info(adjacency)
        model2.load_state_dict(state_dict)
        
        # Check structure info is preserved
        assert torch.allclose(model1.element_adjacency, model2.element_adjacency)
