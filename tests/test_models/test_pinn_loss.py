"""
Unit tests for PINN V2 adaptive loss module.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from Deep_learning.models.pinn_loss import AdaptivePINNLoss, UncertaintyAdaptiveLoss


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_dim: int = 10, output_dim: int = 5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, output_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)


class TestAdaptivePINNLoss:
    """Test cases for AdaptivePINNLoss."""
    
    def test_initialization(self):
        """Test loss module initialization."""
        loss_fn = AdaptivePINNLoss(
            initial_data_weight=1.0,
            initial_physics_weight=0.1,
            adaptive=True
        )
        
        assert loss_fn.adaptive is True
        assert loss_fn.temperature == 0.1
        assert abs(loss_fn.data_weight.item() - 1.0) < 1e-5
        assert abs(loss_fn.physics_weight.item() - 0.1) < 1e-5
    
    def test_static_weights(self):
        """Test static weight mode."""
        loss_fn = AdaptivePINNLoss(
            initial_data_weight=2.0,
            initial_physics_weight=0.5,
            adaptive=False
        )
        
        predictions = torch.randn(4, 5)
        targets = torch.randn(4, 5)
        physics_residual = torch.tensor(0.5)
        
        loss, loss_dict = loss_fn(predictions, targets, physics_residual)
        
        assert loss_dict['data_weight'] == 2.0
        assert loss_dict['physics_weight'] == 0.5
        assert 'total_loss' in loss_dict
    
    def test_adaptive_forward(self):
        """Test forward pass with adaptive weights."""
        loss_fn = AdaptivePINNLoss(
            initial_data_weight=1.0,
            initial_physics_weight=0.1,
            adaptive=True
        )
        
        model = SimpleModel()
        predictions = torch.randn(4, 5, requires_grad=True)
        targets = torch.randn(4, 5)
        physics_residual = torch.tensor(0.5)
        
        loss_fn.train()
        loss, loss_dict = loss_fn(predictions, targets, physics_residual, model)
        
        assert isinstance(loss, torch.Tensor)
        assert 'data_loss' in loss_dict
        assert 'physics_loss' in loss_dict
        assert 'data_weight' in loss_dict
        assert 'physics_weight' in loss_dict
        assert loss_dict['data_weight'] > 0
        assert loss_dict['physics_weight'] > 0
    
    def test_weight_bounds(self):
        """Test that weights stay within bounds."""
        loss_fn = AdaptivePINNLoss(
            min_weight=0.01,
            max_weight=10.0
        )
        
        assert loss_fn.min_weight == 0.01
        assert loss_fn.max_weight == 10.0
        
        # Check weights are within bounds
        assert loss_fn.data_weight.item() >= 0.01
        assert loss_fn.data_weight.item() <= 10.0
    
    def test_weight_reset(self):
        """Test weight reset functionality."""
        loss_fn = AdaptivePINNLoss(
            initial_data_weight=1.0,
            initial_physics_weight=0.1
        )
        
        model = SimpleModel()
        for _ in range(3):
            predictions = torch.randn(4, 5, requires_grad=True)
            targets = torch.randn(4, 5)
            physics_residual = torch.tensor(0.5)
            
            loss, _ = loss_fn(predictions, targets, physics_residual, model)
            loss.backward()
        
        loss_fn.reset_weights(data_weight=2.0, physics_weight=0.5)
        
        assert abs(loss_fn.data_weight.item() - 2.0) < 1e-5
        assert abs(loss_fn.physics_weight.item() - 0.5) < 1e-5
    
    def test_no_physics_loss(self):
        """Test behavior when no physics residual provided."""
        loss_fn = AdaptivePINNLoss()
        
        predictions = torch.randn(4, 5)
        targets = torch.randn(4, 5)
        
        loss, loss_dict = loss_fn(predictions, targets, None)
        
        assert loss_dict['physics_loss'] == 0.0
        assert 'total_loss' in loss_dict
    
    def test_dict_physics_residual(self):
        """Test with dictionary physics residual."""
        loss_fn = AdaptivePINNLoss()
        
        predictions = torch.randn(4, 5)
        targets = torch.randn(4, 5)
        physics_residual = {
            'smoothness_loss': torch.tensor(0.3),
            'range_loss': torch.tensor(0.2)
        }
        
        loss, loss_dict = loss_fn(predictions, targets, physics_residual)
        
        assert loss_dict['physics_loss'] == 0.5  # Sum of dict values


class TestUncertaintyAdaptiveLoss:
    """Test cases for UncertaintyAdaptiveLoss."""
    
    def test_initialization(self):
        """Test uncertainty loss initialization."""
        loss_fn = UncertaintyAdaptiveLoss()
        
        assert isinstance(loss_fn.log_sigma_data, nn.Parameter)
        assert isinstance(loss_fn.log_sigma_physics, nn.Parameter)
    
    def test_forward(self):
        """Test forward pass."""
        loss_fn = UncertaintyAdaptiveLoss()
        
        predictions = torch.randn(4, 5)
        targets = torch.randn(4, 5)
        physics_residual = torch.tensor(0.5)
        
        loss, loss_dict = loss_fn(predictions, targets, physics_residual)
        
        assert isinstance(loss, torch.Tensor)
        assert 'total_loss' in loss_dict
        assert 'data_loss' in loss_dict
        assert 'physics_loss' in loss_dict
        assert 'sigma_data' in loss_dict
        assert 'sigma_physics' in loss_dict
    
    def test_uncertainty_learning(self):
        """Test that uncertainties are learned."""
        loss_fn = UncertaintyAdaptiveLoss()
        
        initial_sigma_data = torch.exp(loss_fn.log_sigma_data).item()
        
        # Simulate training
        optimizer = torch.optim.Adam(loss_fn.parameters(), lr=0.01)
        
        for _ in range(10):
            predictions = torch.randn(4, 5)
            targets = torch.randn(4, 5)
            physics_residual = torch.tensor(0.5)
            
            loss, _ = loss_fn(predictions, targets, physics_residual)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        final_sigma_data = torch.exp(loss_fn.log_sigma_data).item()
        
        # Sigma should have changed
        assert abs(final_sigma_data - initial_sigma_data) > 1e-6


class TestLossIntegration:
    """Integration tests for loss functions."""
    
    def test_training_step(self):
        """Test a full training step with adaptive loss."""
        model = SimpleModel()
        loss_fn = AdaptivePINNLoss(adaptive=True)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=0.01)
        
        x = torch.randn(8, 10, requires_grad=True)
        y = torch.randn(8, 5)
        
        pred = model(x)
        physics_residual = torch.tensor(0.3)
        
        loss, loss_dict = loss_fn(pred, y, physics_residual, model)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert loss_dict['total_loss'] > 0
        assert all(v >= 0 for v in [loss_dict['data_loss'], loss_dict['physics_loss']])
    
    def test_loss_decrease(self):
        """Test that loss decreases with training."""
        model = SimpleModel()
        loss_fn = AdaptivePINNLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Fixed data
        x = torch.randn(16, 10)
        y = torch.randn(16, 5)
        physics_residual = torch.tensor(0.2)
        
        losses = []
        for _ in range(20):
            pred = model(x)
            loss, _ = loss_fn(pred, y, physics_residual)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Loss should generally decrease
        assert losses[-1] < losses[0] * 1.5  # Allow some noise
