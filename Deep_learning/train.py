"""
训练脚本 - 对比LSTM和PINN模型

使用方式:
    python train.py --model lstm --epochs 100
    python train.py --model pinn --epochs 100
    python train.py --model both --epochs 100
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import FEMDataset, split_dataset
from models.gt_model import GTDamagePredictor
from models.pinn_model import PINNDamagePredictor, PINNLoss
from utils.metrics import compute_metrics, EarlyStopping


def build_graph_info(element_conn: np.ndarray, num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建节点邻接矩阵和边索引
    
    Returns:
        adj: (num_nodes, num_nodes)
        edge_index: (num_elements, 2)
    """
    adj = np.eye(num_nodes) # 包含自连接
    edge_index = element_conn.copy()
    
    for n1, n2 in element_conn:
        adj[n1, n2] = 1
        adj[n2, n1] = 1
        
    return adj, edge_index


def build_element_adjacency(element_conn: np.ndarray) -> np.ndarray:
    """构建单元邻接矩阵 (用于 PINN)"""
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


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    model_type: str = 'gt',
    graph_info: Optional[Dict] = None
) -> Dict[str, float]:
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_data_loss = 0
    total_physics_loss = 0
    num_batches = 0
    
    adj = None
    edge_index = None
    if graph_info:
        adj = torch.from_numpy(graph_info['adj']).float().to(device)
        edge_index = torch.from_numpy(graph_info['edge_index']).long().to(device)

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        if model_type == 'pinn':
            outputs = model(x)
            pred = outputs['damage']
            
            physics_losses = model.compute_physics_loss(pred)
            losses = criterion(pred, y, physics_losses)
            loss = losses['total_loss']
            total_data_loss += losses['data_loss'].item()
            total_physics_loss += losses.get('total_physics_loss', torch.tensor(0)).item()
        elif model_type == 'gt':
            pred = model(x, adj, edge_index)
            loss = criterion(pred, y)
            total_data_loss += loss.item()
        else:
            pred = model(x)
            loss = criterion(pred, y)
            total_data_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'data_loss': total_data_loss / num_batches,
        'physics_loss': total_physics_loss / num_batches
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_type: str = 'gt',
    graph_info: Optional[Dict] = None,
    threshold: float = 0.95,
    return_raw: bool = False
) -> Dict[str, float] | Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    adj = None
    edge_index = None
    if graph_info:
        adj = torch.from_numpy(graph_info['adj']).float().to(device)
        edge_index = torch.from_numpy(graph_info['edge_index']).long().to(device)
    
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        
        if model_type == 'pinn':
            outputs = model(x)
            pred = outputs['damage']
            loss = criterion(pred, y)['total_loss']
        elif model_type == 'gt':
            pred = model(x, adj, edge_index)
            loss = criterion(pred, y)
        else:
            pred = model(x)
            loss = criterion(pred, y)
        
        total_loss += loss.item()
        all_preds.append(pred.cpu())
        all_labels.append(y.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    metrics = compute_metrics(all_preds, all_labels, threshold=threshold)
    metrics['loss'] = total_loss / len(val_loader)

    if return_raw:
        return metrics, all_preds.numpy(), all_labels.numpy()
    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    model_type: str = 'gt',
    graph_info: Optional[Dict] = None,
    save_dir: str = 'checkpoints',
    threshold: float = 0.95
) -> Dict[str, list]:
    """训练模型"""
    os.makedirs(save_dir, exist_ok=True)
    
    if model_type == 'pinn':
        criterion = PINNLoss(
            lambda_data=1.0,
            lambda_smooth=config.get('lambda_smooth', 0.1),
            lambda_range=0.01,
            lambda_magnitude=0.05
        )
    else:
        criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    early_stopping = EarlyStopping(patience=config.get('patience', 20))
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_f1': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(config.get('epochs', 100)):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, model_type, graph_info
        )
        
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            model_type,
            graph_info,
            threshold=threshold
        )
        
        scheduler.step(val_metrics['loss'])
        
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_f1'].append(val_metrics['f1'])
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), os.path.join(save_dir, f'{model_type}_best.pth'))
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: train_loss={train_metrics['loss']:.4f}, "
                  f"val_loss={val_metrics['loss']:.4f}, val_mae={val_metrics['mae']:.4f}, "
                  f"val_f1={val_metrics['f1']:.4f}")
        
        if early_stopping(val_metrics['loss']):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Train damage detection models')
    parser.add_argument('--model', type=str, default='gt', choices=['gt', 'pinn', 'both'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--data_path', type=str, default='../dataset/train.npz')
    parser.add_argument('--lambda_smooth', type=float, default=0.1)
    parser.add_argument('--threshold', type=float, default=0.95, help='Damage classification threshold')
    parser.add_argument('--seed', type=int, default=42, help='Global random seed')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, args.data_path)
    
    print(f"Loading data from {data_path}")
    dataset = FEMDataset(data_path, mode='response')
    
    train_set, val_set, test_set = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    sample_x, sample_y = dataset[0]
    # Input x shape for GT: (timesteps, nodes*2) -> Needs reshape to (nodes, timesteps, 2) inside model
    input_dim = sample_x.shape[1]
    output_dim = sample_y.shape[0]
    
    print(f"Sequence length: {sample_x.shape[0]}, Output dim: {output_dim}")
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    
    config = {
        'epochs': args.epochs,
        'lr': args.lr,
        'patience': 20,
        'lambda_smooth': args.lambda_smooth
    }

    results = {}
    prediction_artifacts: Dict[str, np.ndarray] = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join(script_dir, 'checkpoints')
    
    # 图结构信息 (针对 3-Triangle 桁架桥)
    element_conn = np.array([
        [0, 2], [2, 4], [4, 6],
        [1, 3], [3, 5],
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]
    ])
    adj, edge_index = build_graph_info(element_conn, num_nodes=7)
    graph_info = {'adj': adj, 'edge_index': edge_index}

    if args.model in ['gt', 'both']:
        print("\n" + "="*50)
        print("Training Graph Transformer Model")
        print("="*50)
        
        gt_model = GTDamagePredictor(
            num_nodes=7,
            num_elements=output_dim,
            seq_len=sample_x.shape[0],
            node_in_dim=2,
            hidden_dim=args.hidden_dim
        ).to(device)
        
        gt_history = train_model(
            gt_model,
            train_loader,
            val_loader,
            config,
            device,
            'gt',
            graph_info=graph_info,
            save_dir=ckpt_dir,
            threshold=args.threshold
        )

        gt_model.load_state_dict(
            torch.load(os.path.join(ckpt_dir, 'gt_best.pth'), map_location=device)
        )
        gt_test_metrics, gt_test_pred, gt_test_true = evaluate(
            gt_model,
            test_loader,
            nn.MSELoss(),
            device,
            'gt',
            graph_info,
            threshold=args.threshold,
            return_raw=True
        )
        prediction_artifacts['gt_pred'] = gt_test_pred.astype(np.float32)
        prediction_artifacts['true_damage'] = gt_test_true.astype(np.float32)
        
        results['gt'] = {
            'history': gt_history,
            'test_metrics': gt_test_metrics
        }
        print(f"\nGT Test Results: MAE={gt_test_metrics['mae']:.4f}, F1={gt_test_metrics['f1']:.4f}")
    
    if args.model in ['pinn', 'both']:
        print("\n" + "="*50)
        print("Training PINN Model")
        print("="*50)
        
        pinn_model = PINNDamagePredictor(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=output_dim
        ).to(device)
        
        element_conn = np.array([
            [0, 2], [2, 4], [4, 6],
            [1, 3], [3, 5],
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]
        ])
        adjacency = build_element_adjacency(element_conn)
        pinn_model.set_structure_info(adjacency)
        
        pinn_history = train_model(
            pinn_model,
            train_loader,
            val_loader,
            config,
            device,
            'pinn',
            graph_info=None,
            save_dir=ckpt_dir,
            threshold=args.threshold
        )

        pinn_model.load_state_dict(
            torch.load(os.path.join(ckpt_dir, 'pinn_best.pth'), map_location=device)
        )
        criterion = PINNLoss(
            lambda_data=1.0,
            lambda_smooth=config.get('lambda_smooth', 0.1),
            lambda_range=0.01,
            lambda_magnitude=0.05
        )
        pinn_test_metrics, pinn_test_pred, pinn_test_true = evaluate(
            pinn_model,
            test_loader,
            criterion,
            device,
            'pinn',
            threshold=args.threshold,
            return_raw=True
        )
        prediction_artifacts['pinn_pred'] = pinn_test_pred.astype(np.float32)
        if 'true_damage' not in prediction_artifacts:
            prediction_artifacts['true_damage'] = pinn_test_true.astype(np.float32)
        
        results['pinn'] = {
            'history': pinn_history,
            'test_metrics': pinn_test_metrics
        }
        print(f"\nPINN Test Results: MAE={pinn_test_metrics['mae']:.4f}, F1={pinn_test_metrics['f1']:.4f}")
    
    if args.model == 'both':
        print("\n" + "="*50)
        print("Model Comparison")
        print("="*50)
        print(f"{'Metric':<15} {'GT':<15} {'PINN':<15}")
        print("-"*45)
        for metric in ['mae', 'rmse', 'f1', 'iou']:
            gt_val = results['gt']['test_metrics'].get(metric, 0)
            pinn_val = results['pinn']['test_metrics'].get(metric, 0)
            print(f"{metric:<15} {gt_val:<15.4f} {pinn_val:<15.4f}")
    
    results_file = os.path.join(ckpt_dir, f'results_{timestamp}.json')

    serializable_results = {}
    for model_name, model_results in results.items():
        serializable_results[model_name] = {
            'history': {k: [float(v) for v in vals] for k, vals in model_results['history'].items()},
            'test_metrics': {k: float(v) for k, v in model_results['test_metrics'].items()}
        }

    prediction_file: Optional[str] = None
    if prediction_artifacts:
        prediction_file = os.path.join(ckpt_dir, f'predictions_{timestamp}.npz')
        np.savez_compressed(prediction_file, **prediction_artifacts)

    output_payload = {
        'metadata': {
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'threshold': float(args.threshold),
            'seed': int(args.seed),
            'device': str(device),
            'data_path': os.path.abspath(data_path),
            'args': {
                'model': args.model,
                'epochs': int(args.epochs),
                'batch_size': int(args.batch_size),
                'lr': float(args.lr),
                'hidden_dim': int(args.hidden_dim),
                'lambda_smooth': float(args.lambda_smooth)
            },
            'prediction_file': prediction_file
        },
        'models': serializable_results
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {results_file}")
    if prediction_file:
        print(f"Predictions saved to {prediction_file}")


if __name__ == '__main__':
    main()
