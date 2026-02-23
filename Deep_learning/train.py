"""
训练脚本 - 支持 GT / PINN 模型

使用方式:
    python train.py --config ../dataset_config.yaml --model gt --epochs 100
    python train.py --config ../dataset_config.yaml --model pinn --epochs 100
    python train.py --config ../dataset_config.yaml --model both --epochs 100
"""

import os
import sys
import json
import argparse
import random
from functools import partial
from datetime import datetime
from typing import Dict, Optional, Tuple
from contextlib import nullcontext

# Deterministic GEMM workspace for CUDA reproducibility.
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
from models.gt_model import GTDamagePredictor
from models.pinn_model import PINNDamagePredictor, PINNLoss
from utils.metrics import compute_metrics, EarlyStopping
from core.io_parser import YAMLParser


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


def _resolve_path(base_dir: str, value: str) -> str:
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(base_dir, value))


def _load_train_config(config_path: str) -> Dict:
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _set_global_determinism(
    seed: int,
    deterministic: bool = True,
    deterministic_warn_only: bool = False,
    num_threads: int | None = 1,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if num_threads is not None and num_threads > 0:
        torch.set_num_threads(int(num_threads))

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=deterministic_warn_only)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def _resolve_train_device(device_request: str) -> torch.device:
    req = str(device_request).strip().lower()
    if req in {"auto", ""}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if req == "cpu":
        return torch.device("cpu")
    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available.")
        return torch.device(req)
    raise ValueError(f"Unsupported device setting: {device_request}")


def _seed_worker(worker_id: int, base_seed: int) -> None:
    """Top-level worker init function so it is picklable on spawn-based platforms."""
    worker_seed = int(base_seed) + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _build_grad_scaler(amp_enabled: bool, device: torch.device):
    if not (amp_enabled and device.type == 'cuda'):
        return None
    # Prefer torch.amp API (new), fallback to torch.cuda.amp for older PyTorch.
    try:
        return torch.amp.GradScaler('cuda', enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    model_type: str = 'gt',
    graph_info: Optional[Dict] = None,
    amp_enabled: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
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
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        autocast_ctx = (
            torch.autocast(device_type='cuda', dtype=torch.float16)
            if amp_enabled and device.type == 'cuda'
            else nullcontext()
        )
        with autocast_ctx:
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

        if amp_enabled and scaler is not None and device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
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
    return_raw: bool = False,
    amp_enabled: bool = False,
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
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        autocast_ctx = (
            torch.autocast(device_type='cuda', dtype=torch.float16)
            if amp_enabled and device.type == 'cuda'
            else nullcontext()
        )
        with autocast_ctx:
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
    threshold: float = 0.95,
    amp_enabled: bool = False,
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
    scaler = _build_grad_scaler(amp_enabled=amp_enabled, device=device)
    
    for epoch in range(config.get('epochs', 100)):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            model_type,
            graph_info,
            amp_enabled=amp_enabled,
            scaler=scaler,
        )
        
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            model_type,
            graph_info,
            threshold=threshold,
            amp_enabled=amp_enabled,
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
    parser.add_argument('--config', type=str, default=os.path.join(ROOT_DIR, 'dataset_config.yaml'))
    parser.add_argument('--model', type=str, default=None, choices=['gt', 'pinn', 'both'])
    parser.add_argument('--dataset_mode', type=str, default=None, choices=['response', 'load', 'both'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--structure_file', type=str, default=None)
    parser.add_argument('--lambda_smooth', type=float, default=None)
    parser.add_argument('--threshold', type=float, default=None, help='Damage classification threshold')
    parser.add_argument('--seed', type=int, default=None, help='Global random seed')
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--deterministic_warn_only', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--num_threads', type=int, default=None)
    parser.add_argument('--device', type=str, default=None, help='auto/cpu/cuda/cuda:0')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--pin_memory', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--eval_only', action='store_true', help='Skip training and only evaluate saved *_best.pth')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    cfg = _load_train_config(args.config)
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    train_cfg = cfg.get('deep_learning', {}).get('train', {})

    def pick(cli_value, cfg_key: str, default_value):
        if cli_value is not None:
            return cli_value
        return train_cfg.get(cfg_key, default_value)

    model_choice = str(pick(args.model, 'model', 'gt')).lower()
    dataset_mode = str(pick(args.dataset_mode, 'dataset_mode', 'response')).lower()
    epochs = int(pick(args.epochs, 'epochs', 100))
    batch_size = int(pick(args.batch_size, 'batch_size', 32))
    lr = float(pick(args.lr, 'lr', 1e-3))
    hidden_dim = int(pick(args.hidden_dim, 'hidden_dim', 64))
    lambda_smooth = float(pick(args.lambda_smooth, 'lambda_smooth', 0.1))
    threshold = float(pick(args.threshold, 'threshold', 0.95))
    seed = int(pick(args.seed, 'seed', 42))
    patience = int(pick(args.patience, 'patience', 20))
    deterministic = bool(pick(args.deterministic, 'deterministic', True))
    deterministic_warn_only = bool(
        pick(args.deterministic_warn_only, 'deterministic_warn_only', False)
    )
    num_threads = int(pick(args.num_threads, 'num_threads', 1))
    device_request = str(pick(args.device, 'device', 'auto'))
    amp_enabled = bool(pick(args.amp, 'amp', False))
    num_workers = int(pick(args.num_workers, 'num_workers', 0))
    pin_memory_cfg = bool(pick(args.pin_memory, 'pin_memory', True))

    data_path_cfg = args.data_path or cfg.get('output_file', 'dataset/train.npz')
    data_path = _resolve_path(cfg_dir, data_path_cfg)
    structure_path_cfg = args.structure_file or cfg.get('structure_file', 'structure.yaml')
    structure_file = _resolve_path(cfg_dir, structure_path_cfg)

    if args.output_dir is None:
        ckpt_dir = os.path.join(SCRIPT_DIR, 'checkpoints')
    else:
        ckpt_dir = _resolve_path(SCRIPT_DIR, args.output_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    _set_global_determinism(
        seed=seed,
        deterministic=deterministic,
        deterministic_warn_only=deterministic_warn_only,
        num_threads=num_threads,
    )

    device = _resolve_train_device(device_request)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if amp_enabled and device.type != 'cuda':
        print("AMP requested but CUDA is not used. Falling back to FP32 training.")
        amp_enabled = False

    pin_memory = bool(pin_memory_cfg and device.type == 'cuda')

    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"CUDA GPU: {torch.cuda.get_device_name(device)}")
        print(f"CUDA count: {torch.cuda.device_count()}")
    print(f"Config: {args.config}")
    print(f"Loading data from {data_path}")
    print(f"Structure file: {structure_file}")
    print(f"AMP enabled: {amp_enabled}")
    print(f"Eval only: {args.eval_only}")

    dataset = FEMDataset(data_path, mode=dataset_mode)

    train_ratio = float(train_cfg.get('train_ratio', 0.7))
    val_ratio = float(train_cfg.get('val_ratio', 0.15))
    test_ratio = float(train_cfg.get('test_ratio', 0.15))
    train_set, val_set, test_set = split_dataset(
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_loader_generator = torch.Generator().manual_seed(seed)
    worker_init_fn = partial(_seed_worker, base_seed=seed) if num_workers > 0 else None
    train_persistent_workers = bool(num_workers > 0)
    eval_persistent_workers = False
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=train_loader_generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=train_persistent_workers,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=eval_persistent_workers,
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        # Keep test loader single-process for Windows stability at end-of-training evaluation.
        num_workers=0,
        pin_memory=pin_memory,
    )

    sample_x, sample_y = dataset[0]
    input_dim = int(sample_x.shape[1])
    output_dim = int(sample_y.shape[0])
    seq_len = int(sample_x.shape[0])

    structure_data = YAMLParser.load_structure_yaml(structure_file)
    num_nodes = int(structure_data.num_nodes)
    num_elements = int(structure_data.num_elements)
    element_conn = np.asarray(structure_data.element_conn, dtype=np.int64)

    if output_dim != num_elements:
        raise ValueError(
            f"Dataset output_dim={output_dim} does not match structure num_elements={num_elements}"
        )
    if input_dim % num_nodes != 0:
        raise ValueError(
            f"Input dim {input_dim} is not divisible by num_nodes {num_nodes}. "
            "Please align dataset mode and structure."
        )

    node_in_dim = input_dim // num_nodes
    adj, edge_index = build_graph_info(element_conn, num_nodes=num_nodes)
    graph_info = {'adj': adj, 'edge_index': edge_index}

    print(f"Sequence length: {seq_len}, Input dim: {input_dim}, Node in dim: {node_in_dim}, Output dim: {output_dim}")
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    config = {
        'epochs': epochs,
        'lr': lr,
        'patience': patience,
        'lambda_smooth': lambda_smooth
    }

    results = {}
    prediction_artifacts: Dict[str, np.ndarray] = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if model_choice in ['gt', 'both']:
        print("\n" + "="*50)
        print("Training Graph Transformer Model")
        print("="*50)

        gt_model = GTDamagePredictor(
            num_nodes=num_nodes,
            num_elements=output_dim,
            seq_len=seq_len,
            node_in_dim=node_in_dim,
            hidden_dim=hidden_dim
        ).to(device)

        gt_ckpt = os.path.join(ckpt_dir, 'gt_best.pth')
        if args.eval_only:
            if not os.path.exists(gt_ckpt):
                raise FileNotFoundError(f"Checkpoint not found for eval_only mode: {gt_ckpt}")
            gt_history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_f1': []}
        else:
            gt_history = train_model(
                gt_model,
                train_loader,
                val_loader,
                config,
                device,
                'gt',
                graph_info=graph_info,
                save_dir=ckpt_dir,
                threshold=threshold,
                amp_enabled=amp_enabled,
            )

        gt_model.load_state_dict(torch.load(gt_ckpt, map_location=device))
        gt_test_metrics, gt_test_pred, gt_test_true = evaluate(
            gt_model,
            test_loader,
            nn.MSELoss(),
            device,
            'gt',
            graph_info,
            threshold=threshold,
            return_raw=True,
            amp_enabled=amp_enabled,
        )
        prediction_artifacts['gt_pred'] = gt_test_pred.astype(np.float32)
        prediction_artifacts['true_damage'] = gt_test_true.astype(np.float32)

        results['gt'] = {
            'history': gt_history,
            'test_metrics': gt_test_metrics
        }
        print(f"\nGT Test Results: MAE={gt_test_metrics['mae']:.4f}, F1={gt_test_metrics['f1']:.4f}")

    if model_choice in ['pinn', 'both']:
        print("\n" + "="*50)
        print("Training PINN Model")
        print("="*50)

        pinn_model = PINNDamagePredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        ).to(device)

        adjacency = build_element_adjacency(element_conn)
        pinn_model.set_structure_info(adjacency)

        pinn_ckpt = os.path.join(ckpt_dir, 'pinn_best.pth')
        if args.eval_only:
            if not os.path.exists(pinn_ckpt):
                raise FileNotFoundError(f"Checkpoint not found for eval_only mode: {pinn_ckpt}")
            pinn_history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_f1': []}
        else:
            pinn_history = train_model(
                pinn_model,
                train_loader,
                val_loader,
                config,
                device,
                'pinn',
                graph_info=None,
                save_dir=ckpt_dir,
                threshold=threshold,
                amp_enabled=amp_enabled,
            )

        pinn_model.load_state_dict(torch.load(pinn_ckpt, map_location=device))
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
            threshold=threshold,
            return_raw=True,
            amp_enabled=amp_enabled,
        )
        prediction_artifacts['pinn_pred'] = pinn_test_pred.astype(np.float32)
        if 'true_damage' not in prediction_artifacts:
            prediction_artifacts['true_damage'] = pinn_test_true.astype(np.float32)

        results['pinn'] = {
            'history': pinn_history,
            'test_metrics': pinn_test_metrics
        }
        print(f"\nPINN Test Results: MAE={pinn_test_metrics['mae']:.4f}, F1={pinn_test_metrics['f1']:.4f}")

    if model_choice == 'both':
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
            'threshold': float(threshold),
            'seed': int(seed),
            'device': str(device),
            'config_path': os.path.abspath(args.config),
            'data_path': os.path.abspath(data_path),
            'structure_file': os.path.abspath(structure_file),
            'dataset_mode': dataset_mode,
            'deterministic': deterministic,
            'deterministic_warn_only': deterministic_warn_only,
            'num_threads': int(num_threads),
            'device_request': device_request,
            'amp_enabled': bool(amp_enabled),
            'num_workers': int(num_workers),
            'pin_memory': bool(pin_memory),
            'args': {
                'model': model_choice,
                'epochs': int(epochs),
                'batch_size': int(batch_size),
                'lr': float(lr),
                'hidden_dim': int(hidden_dim),
                'lambda_smooth': float(lambda_smooth)
            },
            'eval_only': bool(args.eval_only),
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
