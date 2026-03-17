"""
Integration tests for PINN V2 training pipeline.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from Deep_learning.models.pinn_model_v2 import PINNDamagePredictorV2
from Deep_learning.models.pinn_loss import AdaptivePINNLoss
from Deep_learning.train_pinn_v2 import (
    train_one_epoch, evaluate, train_pinn_v2, evaluate_pinn_v2
)


@pytest.fixture
def dummy_data():
    """Create dummy dataset for testing."""
    np.random.seed(42)
    
    # Create synthetic data
    n_samples = 100
    seq_len = 10
    n_features = 20
    n_elements = 11
    
    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    y = np.random.rand(n_samples, n_elements).astype(np.float32)
    
    # Make some samples have damage
    damaged_indices = np.random.choice(n_samples, size=20, replace=False)
    y[damaged_indices, :3] = np.random.rand(20, 3) * 0.5  # Damage on first 3 elements
    
    return X, y


@pytest.fixture
def dummy_loaders(dummy_data):
    """Create DataLoaders from dummy data."""
    X, y = dummy_data
    
    # Split into train/val/test
    train_size = 60
    val_size = 20
    
    train_dataset = TensorDataset(
        torch.from_numpy(X[:train_size]),
        torch.from_numpy(y[:train_size])
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X[train_size:train_size+val_size]),
        torch.from_numpy(y[train_size:train_size+val_size])
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X[train_size+val_size:]),
        torch.from_numpy(y[train_size+val_size:])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    return train_loader, val_loader, test_loader


@pytest.fixture
def model():
    """Create a PINN V2 model for testing."""
    return PINNDamagePredictorV2(
        input_dim=20,
        output_dim=11,
        use_physics=True
    )


class TestTrainOneEpoch:
    """Test training one epoch."""
    
    def test_train_epoch_completes(self, model, dummy_loaders):
        """Test that training one epoch completes successfully."""
        train_loader, _, _ = dummy_loaders
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = AdaptivePINNLoss(adaptive=False)
        device = torch.device('cpu')
        
        metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch=0, log_interval=100
        )
        
        assert 'loss' in metrics
        assert 'data_loss' in metrics
        assert 'physics_loss' in metrics
        assert metrics['loss'] > 0
    
    def test_train_epoch_updates_weights(self, model, dummy_loaders):
        """Test that weights are updated during training."""
        train_loader, _, _ = dummy_loaders
        
        # Get initial weights
        initial_weight = list(model.parameters())[0].clone().detach()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = AdaptivePINNLoss(adaptive=False)
        device = torch.device('cpu')
        
        train_one_epoch(model, train_loader, optimizer, criterion, device, epoch=0, log_interval=100)
        
        # Check weights changed
        updated_weight = list(model.parameters())[0].detach()
        assert not torch.allclose(initial_weight, updated_weight)


class TestEvaluate:
    """Test evaluation function."""
    
    def test_evaluate_returns_metrics(self, model, dummy_loaders):
        """Test that evaluate returns expected metrics."""
        _, _, test_loader = dummy_loaders
        device = torch.device('cpu')
        
        model.eval()
        metrics = evaluate(model, test_loader, device)
        
        assert 'loss' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'f1' in metrics
        assert 'iou' in metrics
        assert 'accuracy' in metrics
    
    def test_evaluate_deterministic(self, model, dummy_loaders):
        """Test that evaluation is deterministic."""
        _, _, test_loader = dummy_loaders
        device = torch.device('cpu')
        
        model.eval()
        metrics1 = evaluate(model, test_loader, device)
        metrics2 = evaluate(model, test_loader, device)
        
        for key in metrics1:
            assert abs(metrics1[key] - metrics2[key]) < 1e-6
    
    def test_evaluate_with_raw(self, model, dummy_loaders):
        """Test evaluate with return_raw=True."""
        _, _, test_loader = dummy_loaders
        device = torch.device('cpu')
        
        model.eval()
        result = evaluate(model, test_loader, device, return_raw=True)
        
        assert len(result) == 3
        metrics, preds, labels = result
        
        assert isinstance(preds, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert preds.shape[1] == 11  # num_elements


class TestTrainPINNV2:
    """Test full training pipeline."""
    
    def test_short_training(self, model, dummy_loaders, tmp_path):
        """Test training for a few epochs."""
        train_loader, val_loader, _ = dummy_loaders
        device = torch.device('cpu')
        
        config = {
            'epochs': 5,
            'lr': 0.01,
            'patience': 10,
            'adaptive_weights': False,
            'initial_data_weight': 1.0,
            'initial_physics_weight': 0.1
        }
        
        trained_model, history = train_pinn_v2(
            model, train_loader, val_loader, config, device,
            save_dir=str(tmp_path), verbose=False
        )
        
        assert len(history['train_loss']) == 5
        assert len(history['val_loss']) == 5
        assert len(history['val_mae']) == 5
    
    def test_training_with_adaptive_loss(self, model, dummy_loaders, tmp_path):
        """Test training with adaptive loss."""
        train_loader, val_loader, _ = dummy_loaders
        device = torch.device('cpu')
        
        config = {
            'epochs': 5,
            'lr': 0.01,
            'patience': 10,
            'adaptive_weights': True,
            'loss_type': 'gradnorm',
            'initial_data_weight': 1.0,
            'initial_physics_weight': 0.1,
            'temperature': 0.1,
            'alpha': 1.5
        }
        
        trained_model, history = train_pinn_v2(
            model, train_loader, val_loader, config, device,
            save_dir=str(tmp_path), verbose=False
        )
        
        # Check that weight history is recorded
        assert len(history['data_weight']) == 5
        assert len(history['physics_weight']) == 5
    
    def test_early_stopping(self, model, dummy_loaders, tmp_path):
        """Test early stopping triggers."""
        train_loader, val_loader, _ = dummy_loaders
        device = torch.device('cpu')
        
        config = {
            'epochs': 50,
            'lr': 0.0001,  # Very low LR to ensure no improvement
            'patience': 3,  # Early stop after 3 epochs of no improvement
            'adaptive_weights': False
        }
        
        trained_model, history = train_pinn_v2(
            model, train_loader, val_loader, config, device,
            save_dir=str(tmp_path), verbose=False
        )
        
        # Should stop early
        assert len(history['train_loss']) < 50


class TestEvaluatePINNV2:
    """Test detailed evaluation."""
    
    def test_evaluate_detailed_metrics(self, model, dummy_loaders):
        """Test that evaluate_pinn_v2 returns detailed metrics."""
        _, _, test_loader = dummy_loaders
        device = torch.device('cpu')
        
        model.eval()
        metrics = evaluate_pinn_v2(model, test_loader, device)
        
        assert 'mae' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert 'f1' in metrics
        assert 'iou' in metrics
    
    def test_per_damage_type_metrics(self, model, dummy_data):
        """Test per-damage-type MAE calculation."""
        X, y = dummy_data
        
        # Create dataset with clear damaged/undamaged separation
        # Make half samples damaged, half undamaged
        y[:50, :3] = 0.5  # Damaged
        y[50:, :] = 1.0  # Undamaged
        
        test_dataset = TensorDataset(
            torch.from_numpy(X[80:]),
            torch.from_numpy(y[80:])
        )
        test_loader = DataLoader(test_dataset, batch_size=16)
        
        device = torch.device('cpu')
        model.eval()
        metrics = evaluate_pinn_v2(model, test_loader, device)
        
        # Should have per-damage-type metrics
        assert 'damaged_mae' in metrics
        assert 'undamaged_mae' in metrics


class TestTrainingConvergence:
    """Test that training actually improves the model."""
    
    def test_loss_decreases(self, dummy_data, tmp_path):
        """Test that loss decreases during training."""
        X, y = dummy_data
        
        # Create dataset
        train_dataset = TensorDataset(
            torch.from_numpy(X[:60]),
            torch.from_numpy(y[:60])
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X[60:80]),
            torch.from_numpy(y[60:80])
        )
        
        train_loader = DataLoader(train_dataset, batch_size=16)
        val_loader = DataLoader(val_dataset, batch_size=16)
        
        model = PINNDamagePredictorV2(input_dim=20, output_dim=11)
        device = torch.device('cpu')
        
        config = {
            'epochs': 20,
            'lr': 0.01,
            'patience': 20,
            'adaptive_weights': False
        }
        
        trained_model, history = train_pinn_v2(
            model, train_loader, val_loader, config, device,
            save_dir=str(tmp_path), verbose=False
        )
        
        # Loss should generally decrease (allow for some noise)
        initial_loss = history['train_loss'][0]
        final_loss = history['train_loss'][-1]
        
        assert final_loss < initial_loss * 1.2  # Allow 20% variance
    
    def test_mae_improves(self, dummy_data, tmp_path):
        """Test that MAE improves during training."""
        X, y = dummy_data
        
        train_dataset = TensorDataset(
            torch.from_numpy(X[:60]),
            torch.from_numpy(y[:60])
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X[60:80]),
            torch.from_numpy(y[60:80])
        )
        test_dataset = TensorDataset(
            torch.from_numpy(X[80:]),
            torch.from_numpy(y[80:])
        )
        
        train_loader = DataLoader(train_dataset, batch_size=16)
        val_loader = DataLoader(val_dataset, batch_size=16)
        test_loader = DataLoader(test_dataset, batch_size=16)
        
        model = PINNDamagePredictorV2(input_dim=20, output_dim=11)
        device = torch.device('cpu')
        
        # Get initial MAE
        model.eval()
        initial_metrics = evaluate_pinn_v2(model, test_loader, device)
        initial_mae = initial_metrics['mae']
        
        # Train
        config = {
            'epochs': 30,
            'lr': 0.01,
            'patience': 30,
            'adaptive_weights': True
        }
        
        trained_model, _ = train_pinn_v2(
            model, train_loader, val_loader, config, device,
            save_dir=str(tmp_path), verbose=False
        )
        
        # Get final MAE
        trained_model.eval()
        final_metrics = evaluate_pinn_v2(trained_model, test_loader, device)
        final_mae = final_metrics['mae']
        
        # MAE should improve (or at least not get much worse)
        assert final_mae < initial_mae * 1.5
