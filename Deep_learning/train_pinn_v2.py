"""
Training script for PINN V2 with adaptive loss weighting.

Usage:
    python train_pinn_v2.py --config configs/pinn_v2.yaml
    python train_pinn_v2.py --epochs 50 --adaptive_weights
    python train_pinn_v2.py --grid_search
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from contextlib import nullcontext

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
PYFEM_DIR = os.path.join(ROOT_DIR, "PyFEM_Dynamics")
for path in (SCRIPT_DIR, ROOT_DIR, PYFEM_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from data.dataset import FEMDataset, split_dataset
from models.pinn_model_v2 import PINNDamagePredictorV2
from models.pinn_loss import AdaptivePINNLoss, UncertaintyAdaptiveLoss
from utils.metrics import compute_metrics, EarlyStopping
from core.io_parser import YAMLParser


def build_element_adjacency(element_conn: np.ndarray) -> np.ndarray:
    """Build element adjacency matrix for PINN physics constraints."""
    num_elem = element_conn.shape[0]
    adjacency = np.zeros((num_elem, num_elem))
    
    node_to_elements = {}
    for i, (n1, n2) in enumerate(element_conn):
        for node in [n1, n2]:
            if node not in node_to_elements:
                node_to_elements[node] = []
            node_to_elements[node].append(i)
    
    for elements in node_to_elements.values():
        for i in elements:
            for j in elements:
                if i != j:
                    adjacency[i, j] = 1
    
    return adjacency


def _resolve_path(base_dir: str, value: str) -> str:
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(base_dir, value))


def _set_global_determinism(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int = 10
) -> Dict[str, float]:
    """Train one epoch with adaptive loss."""
    model.train()
    
    total_loss = 0
    total_data_loss = 0
    total_physics_loss = 0
    num_batches = 0
    
    # Track weight history
    data_weights = []
    physics_weights = []
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        damage_pred = model(x)
        
        # Compute physics losses
        physics_losses = model.physics_loss(damage_pred)
        physics_residual = physics_losses['total_physics_loss']
        
        # Compute adaptive loss
        if isinstance(criterion, AdaptivePINNLoss):
            loss, loss_dict = criterion(damage_pred, y, physics_residual, model=model)
            data_weights.append(loss_dict['data_weight'])
            physics_weights.append(loss_dict['physics_weight'])
        elif isinstance(criterion, UncertaintyAdaptiveLoss):
            loss, loss_dict = criterion(damage_pred, y, physics_residual)
        else:
            # Standard loss
            data_loss = torch.nn.functional.mse_loss(damage_pred, y)
            physics_loss = physics_residual
            loss = data_loss + 0.1 * physics_loss
            loss_dict = {
                'total_loss': loss.item(),
                'data_loss': data_loss.item(),
                'physics_loss': physics_loss.item()
            }
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss_dict['total_loss']
        total_data_loss += loss_dict['data_loss']
        total_physics_loss += loss_dict['physics_loss']
        num_batches += 1
        
        if batch_idx % log_interval == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}: "
                  f"loss={loss_dict['total_loss']:.4f}, "
                  f"data={loss_dict['data_loss']:.4f}, "
                  f"physics={loss_dict['physics_loss']:.4f}")
    
    metrics = {
        'loss': total_loss / num_batches,
        'data_loss': total_data_loss / num_batches,
        'physics_loss': total_physics_loss / num_batches
    }
    
    if data_weights:
        metrics['avg_data_weight'] = np.mean(data_weights)
        metrics['avg_physics_weight'] = np.mean(physics_weights)
    
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    return_raw: bool = False
) -> Dict[str, float] | Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate model on validation/test set."""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        damage_pred = model(x)
        
        # Compute losses
        physics_losses = model.physics_loss(damage_pred)
        data_loss = torch.nn.functional.mse_loss(damage_pred, y)
        physics_loss = physics_losses['total_physics_loss']
        loss = data_loss + 0.1 * physics_loss
        
        total_loss += loss.item()
        all_preds.append(damage_pred.cpu())
        all_labels.append(y.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    metrics = compute_metrics(all_preds, all_labels, threshold=0.95)
    metrics['loss'] = total_loss / len(val_loader)
    
    if return_raw:
        return metrics, all_preds.numpy(), all_labels.numpy()
    return metrics


def train_pinn_v2(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: torch.device,
    save_dir: str = 'checkpoints',
    verbose: bool = True
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train PINN V2 with adaptive loss weighting.
    
    Returns:
        model: Trained model
        history: Dictionary with training history
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup loss function
    adaptive = config.get('adaptive_weights', True)
    loss_type = config.get('loss_type', 'gradnorm')  # 'gradnorm' or 'uncertainty'
    
    if loss_type == 'uncertainty':
        criterion = UncertaintyAdaptiveLoss()
    else:
        criterion = AdaptivePINNLoss(
            initial_data_weight=config.get('initial_data_weight', 1.0),
            initial_physics_weight=config.get('initial_physics_weight', 0.1),
            adaptive=adaptive,
            temperature=config.get('temperature', 0.1),
            alpha=config.get('alpha', 1.5)
        )
    
    criterion = criterion.to(device)
    
    # Setup optimizer
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Setup scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.get('patience', 30))
    
    # Training history
    history = {
        'train_loss': [],
        'train_data_loss': [],
        'train_physics_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': [],
        'val_f1': [],
        'data_weight': [],
        'physics_weight': []
    }
    
    best_val_mae = float('inf')
    epochs = config.get('epochs', 100)
    
    if verbose:
        print(f"\nTraining PINN V2 ({'adaptive' if adaptive else 'static'} loss)")
        print(f"Epochs: {epochs}, LR: {lr}, Adaptive: {adaptive}")
        print("="*60)
    
    for epoch in range(epochs):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        
        # Update scheduler
        scheduler.step(epoch + train_metrics['loss'])
        
        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['train_data_loss'].append(train_metrics['data_loss'])
        history['train_physics_loss'].append(train_metrics['physics_loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_f1'].append(val_metrics['f1'])
        
        if 'avg_data_weight' in train_metrics:
            history['data_weight'].append(train_metrics['avg_data_weight'])
            history['physics_weight'].append(train_metrics['avg_physics_weight'])
        
        # Save best model
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_mae': val_metrics['mae'],
                'config': config
            }, os.path.join(save_dir, 'pinn_v2_best.pth'))
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train: loss={train_metrics['loss']:.4f}, "
                  f"data={train_metrics['data_loss']:.4f}, "
                  f"physics={train_metrics['physics_loss']:.4f}")
            print(f"  Val: loss={val_metrics['loss']:.4f}, "
                  f"MAE={val_metrics['mae']:.4f}, "
                  f"F1={val_metrics['f1']:.4f}")
            if 'avg_data_weight' in train_metrics:
                print(f"  Weights: data={train_metrics['avg_data_weight']:.4f}, "
                      f"physics={train_metrics['avg_physics_weight']:.4f}")
        
        # Early stopping
        if early_stopping(val_metrics['loss']):
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    checkpoint = torch.load(os.path.join(save_dir, 'pinn_v2_best.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history


def evaluate_pinn_v2(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate PINN V2 with detailed metrics.
    
    Returns:
        Dictionary with MAE, MSE, RMSE, R2, F1, IoU
    """
    metrics, preds, labels = evaluate(model, test_loader, device, return_raw=True)
    
    # Compute per-damage-type metrics
    damaged_mask = labels < 0.95
    undamaged_mask = labels >= 0.95
    
    if damaged_mask.any():
        damaged_mae = np.abs(preds[damaged_mask] - labels[damaged_mask]).mean()
        metrics['damaged_mae'] = float(damaged_mae)
    
    if undamaged_mask.any():
        undamaged_mae = np.abs(preds[undamaged_mask] - labels[undamaged_mask]).mean()
        metrics['undamaged_mae'] = float(undamaged_mae)
    
    # Compute R2
    ss_res = ((preds - labels) ** 2).sum()
    ss_tot = ((labels - labels.mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    metrics['r2'] = float(r2)
    
    return metrics


def run_hyperparameter_search(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: int,
    output_dim: int,
    device: torch.device,
    save_dir: str
) -> Dict:
    """Run grid search over hyperparameters."""
    
    param_grid = {
        'initial_data_weight': [1.0, 2.0],
        'initial_physics_weight': [0.05, 0.1, 0.2],
        'temperature': [0.05, 0.1, 0.2],
        'alpha': [1.0, 1.5, 2.0],
        'lr': [1e-3, 5e-4],
        'hidden_dims': [[256, 256, 128], [256, 128, 128]]
    }
    
    results = []
    
    # For simplicity, do a few key combinations
    configs = [
        {'initial_data_weight': 1.0, 'initial_physics_weight': 0.1, 'temperature': 0.1, 'alpha': 1.5, 'lr': 1e-3, 'hidden_dims': [256, 256, 128]},
        {'initial_data_weight': 2.0, 'initial_physics_weight': 0.05, 'temperature': 0.05, 'alpha': 1.5, 'lr': 5e-4, 'hidden_dims': [256, 256, 128]},
        {'initial_data_weight': 1.0, 'initial_physics_weight': 0.2, 'temperature': 0.2, 'alpha': 2.0, 'lr': 1e-3, 'hidden_dims': [256, 128, 128]},
        {'initial_data_weight': 1.0, 'initial_physics_weight': 0.1, 'temperature': 0.1, 'alpha': 1.0, 'lr': 1e-3, 'hidden_dims': [256, 256, 128]},
    ]
    
    print("\n" + "="*60)
    print("Hyperparameter Search")
    print("="*60)
    
    for i, config in enumerate(configs):
        print(f"\nConfig {i+1}/{len(configs)}: {config}")
        
        # Create model
        model = PINNDamagePredictorV2(
            input_dim=input_dim,
            hidden_dims=config['hidden_dims'],
            output_dim=output_dim,
            use_physics=True
        ).to(device)
        
        # Train
        train_config = {
            'epochs': 30,  # Shorter for grid search
            'lr': config['lr'],
            'patience': 10,
            'adaptive_weights': True,
            'initial_data_weight': config['initial_data_weight'],
            'initial_physics_weight': config['initial_physics_weight'],
            'temperature': config['temperature'],
            'alpha': config['alpha']
        }
        
        model, history = train_pinn_v2(
            model, train_loader, val_loader, train_config, device, save_dir, verbose=False
        )
        
        # Evaluate
        test_metrics = evaluate_pinn_v2(model, test_loader, device)
        
        result = {
            'config': config,
            'val_mae': min(history['val_mae']),
            'test_mae': test_metrics['mae'],
            'test_f1': test_metrics['f1'],
            'epochs_trained': len(history['train_loss'])
        }
        results.append(result)
        
        print(f"  Val MAE: {result['val_mae']:.4f}, Test MAE: {result['test_mae']:.4f}")
    
    # Find best config
    best_idx = min(range(len(results)), key=lambda i: results[i]['test_mae'])
    best_config = results[best_idx]['config']
    
    print(f"\nBest Config (Test MAE: {results[best_idx]['test_mae']:.4f}):")
    print(f"  {best_config}")
    
    return {
        'all_results': results,
        'best_config': best_config,
        'best_mae': results[best_idx]['test_mae']
    }


def main():
    parser = argparse.ArgumentParser(description='Train PINN V2 with adaptive loss')
    parser.add_argument('--config', type=str, default=os.path.join(SCRIPT_DIR, 'configs', 'pinn_v2.yaml'))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data_path', type=str, default=os.path.join(ROOT_DIR, 'dataset', 'train.npz'))
    parser.add_argument('--structure_file', type=str, default=os.path.join(ROOT_DIR, 'structure.yaml'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--adaptive_weights', action='store_true', default=True)
    parser.add_argument('--static_weights', action='store_true')
    parser.add_argument('--grid_search', action='store_true')
    parser.add_argument('--output_dir', type=str, default=os.path.join(SCRIPT_DIR, 'checkpoints'))
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Device: {device}")
    
    # Set seed
    _set_global_determinism(args.seed)
    
    # Load data
    print(f"Loading dataset from {args.data_path}")
    dataset = FEMDataset(args.data_path, mode='response')
    train_set, val_set, test_set = split_dataset(dataset, seed=args.seed)
    dataset.fit_normalization(train_set.indices)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    # Get dimensions
    sample_x, sample_y = dataset[0]
    input_dim = sample_x.shape[1]
    output_dim = sample_y.shape[0]
    
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    
    # Load structure info
    structure_data = YAMLParser.load_structure_yaml(args.structure_file)
    element_conn = np.asarray(structure_data.element_conn, dtype=np.int64)
    adjacency = build_element_adjacency(element_conn)
    
    # Hyperparameter search or single training
    if args.grid_search:
        search_results = run_hyperparameter_search(
            train_loader, val_loader, test_loader,
            input_dim, output_dim, device, args.output_dir
        )
        
        # Save results
        os.makedirs('experiments', exist_ok=True)
        with open('experiments/grid_search_results.json', 'w') as f:
            json.dump(search_results, f, indent=2)
        
        print("\nGrid search results saved to experiments/grid_search_results.json")
        return
    
    # Load config if exists
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'patience': 30,
        'adaptive_weights': not args.static_weights,
        'loss_type': 'gradnorm',
        'initial_data_weight': 1.0,
        'initial_physics_weight': 0.1,
        'temperature': 0.1,
        'alpha': 1.5
    }
    
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_config = yaml.safe_load(f) or {}
            config.update(file_config.get('train', {}))
    
    # Create model
    model = PINNDamagePredictorV2(
        input_dim=input_dim,
        output_dim=output_dim,
        use_physics=True
    ).to(device)
    
    model.set_structure_info(adjacency)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    model, history = train_pinn_v2(
        model, train_loader, val_loader, config, device, args.output_dir
    )

    checkpoint_path = os.path.join(args.output_dir, 'pinn_v2_best.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint['model_name'] = 'pinn_v2'
    checkpoint['model_args'] = {
        'input_dim': int(input_dim),
        'hidden_dims': list(config.get('hidden_dims', [256, 256, 128])),
        'output_dim': int(output_dim),
        'num_residual_blocks': int(config.get('num_residual_blocks', 2)),
        'dropout': float(config.get('dropout', 0.3)),
        'use_physics': bool(config.get('use_physics', True)),
    }
    checkpoint['preprocessing'] = {
        key: value.tolist()
        for key, value in dataset.get_normalization_stats().items()
    }
    checkpoint['training_task'] = str(config.get('task_mode', 'final_state'))
    checkpoint['feature_mode'] = 'response'
    torch.save(checkpoint, checkpoint_path)
    
    # Final evaluation
    test_metrics = evaluate_pinn_v2(model, test_loader, device)
    
    print("\n" + "="*60)
    print("Final Test Results")
    print("="*60)
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save results
    results = {
        'config': config,
        'history': history,
        'test_metrics': test_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    os.makedirs('experiments', exist_ok=True)
    with open('experiments/pinn_v2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to experiments/pinn_v2_results.json")
    print(f"Best checkpoint: {os.path.join(args.output_dir, 'pinn_v2_best.pth')}")


if __name__ == '__main__':
    main()
