"""
Training entry point for GT, legacy PINN, and PINN V2 models.
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from functools import partial
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

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
from models.pinn_model_v2 import PINNDamagePredictorV2
from utils.metrics import compute_metrics, EarlyStopping
from core.io_parser import YAMLParser
from train_pinn_v2 import (
    train_pinn_v2 as train_pinn_v2_model,
    evaluate as evaluate_pinn_v2_loader,
)


LEGACY_PINN_ALIASES = {"pinn", "legacy_pinn"}


def build_graph_info(element_conn: np.ndarray, num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    adj = np.eye(num_nodes)
    edge_index = element_conn.copy()
    for n1, n2 in element_conn:
        adj[n1, n2] = 1
        adj[n2, n1] = 1
    return adj, edge_index


def build_element_adjacency(element_conn: np.ndarray) -> np.ndarray:
    num_elem = element_conn.shape[0]
    adjacency = np.zeros((num_elem, num_elem))
    node_to_elements: Dict[int, List[int]] = {}
    for i, (n1, n2) in enumerate(element_conn):
        node_to_elements.setdefault(int(n1), []).append(i)
        node_to_elements.setdefault(int(n2), []).append(i)
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


def _load_train_config(config_path: str) -> Dict[str, Any]:
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
    worker_seed = int(base_seed) + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _build_grad_scaler(amp_enabled: bool, device: torch.device):
    if not (amp_enabled and device.type == 'cuda'):
        return None
    try:
        return torch.amp.GradScaler('cuda', enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def _normalize_model_choice(model_choice: str) -> str:
    choice = str(model_choice).lower()
    if choice in LEGACY_PINN_ALIASES:
        return "legacy_pinn"
    return choice


def _mask_future_steps(x: torch.Tensor, visible_steps: int | torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        return x
    if isinstance(visible_steps, int):
        visible = torch.full((x.size(0),), int(visible_steps), device=x.device, dtype=torch.long)
    else:
        visible = visible_steps.to(device=x.device, dtype=torch.long)
    time_index = torch.arange(x.size(1), device=x.device).unsqueeze(0)
    mask = (time_index < visible.unsqueeze(1)).unsqueeze(-1)
    return x * mask


def _sample_random_prefix_lengths(x: torch.Tensor, min_visible_steps: int) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError("random prefix masking requires time-series input.")
    low = max(1, int(min_visible_steps))
    high = x.size(1) + 1
    return torch.randint(low=low, high=high, size=(x.size(0),), device=x.device)


def _serialize_numpy_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    serialized: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        else:
            serialized[key] = value
    return serialized


def _make_checkpoint_payload(
    model: nn.Module,
    model_name: str,
    model_args: Dict[str, Any],
    preprocessing: Dict[str, np.ndarray],
    training_task: str,
    feature_mode: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'model_args': dict(model_args),
        'preprocessing': _serialize_numpy_dict(preprocessing),
        'training_task': training_task,
        'feature_mode': feature_mode,
    }
    if extra_metadata:
        payload.update(extra_metadata)
    return payload


def _load_checkpoint_payload(checkpoint_path: str, map_location: torch.device | str) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint
    return {'model_state_dict': checkpoint}


def _infer_gt_hidden_dim(state_dict: Dict[str, torch.Tensor]) -> int:
    return int(state_dict["node_encoder.2.weight"].shape[0])


def _infer_gt_node_in_dim(state_dict: Dict[str, torch.Tensor]) -> int:
    return int(state_dict["node_encoder.0.weight"].shape[1])


def _infer_legacy_pinn_hidden_dim(state_dict: Dict[str, torch.Tensor]) -> int:
    return int(state_dict["encoder.0.weight"].shape[0])


def _infer_legacy_pinn_input_dim(state_dict: Dict[str, torch.Tensor]) -> int:
    return int(state_dict["encoder.0.weight"].shape[1])


def _infer_legacy_pinn_num_layers(state_dict: Dict[str, torch.Tensor]) -> int:
    return len([k for k in state_dict.keys() if k.startswith("encoder.") and k.endswith(".weight")])


def _forward_model(
    model: nn.Module,
    x: torch.Tensor,
    model_type: str,
    graph_info: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    if model_type == 'gt':
        if graph_info is None:
            raise ValueError("graph_info is required for GT model.")
        return model(x, graph_info['adj'], graph_info['edge_index'])
    if model_type == 'legacy_pinn':
        return model(x)['damage']
    if model_type == 'pinn_v2':
        return model(x)
    raise ValueError(f"Unsupported model_type: {model_type}")


def _compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    model: nn.Module,
    criterion: nn.Module,
    model_type: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if model_type == 'legacy_pinn':
        physics_losses = model.compute_physics_loss(pred)
        losses = criterion(pred, target, physics_losses)
        return losses['total_loss'], {
            'data_loss': float(losses['data_loss'].item()),
            'physics_loss': float(losses.get('total_physics_loss', torch.tensor(0.0)).item()),
        }

    loss = criterion(pred, target)
    return loss, {
        'data_loss': float(loss.item()),
        'physics_loss': 0.0,
    }


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    model_type: str = 'gt',
    graph_info: Optional[Dict[str, np.ndarray]] = None,
    amp_enabled: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    task_mode: str = 'final_state',
    online_prefix_min_visible: int = 1,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_data_loss = 0.0
    total_physics_loss = 0.0
    num_batches = 0

    graph_tensors = None
    if graph_info:
        graph_tensors = {
            'adj': torch.from_numpy(graph_info['adj']).float().to(device),
            'edge_index': torch.from_numpy(graph_info['edge_index']).long().to(device),
        }

    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if task_mode == 'online_prefix':
            visible_steps = _sample_random_prefix_lengths(x, online_prefix_min_visible)
            x = _mask_future_steps(x, visible_steps)
        optimizer.zero_grad(set_to_none=True)

        autocast_ctx = (
            torch.autocast(device_type='cuda', dtype=torch.float16)
            if amp_enabled and device.type == 'cuda'
            else nullcontext()
        )
        with autocast_ctx:
            pred = _forward_model(model, x, model_type, graph_tensors)
            loss, loss_parts = _compute_loss(pred, y, model, criterion, model_type)

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

        total_loss += float(loss.item())
        total_data_loss += loss_parts['data_loss']
        total_physics_loss += loss_parts['physics_loss']
        num_batches += 1

    return {
        'loss': total_loss / max(1, num_batches),
        'data_loss': total_data_loss / max(1, num_batches),
        'physics_loss': total_physics_loss / max(1, num_batches),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_type: str = 'gt',
    graph_info: Optional[Dict[str, np.ndarray]] = None,
    threshold: float = 0.95,
    return_raw: bool = False,
    amp_enabled: bool = False,
    visible_steps: Optional[int] = None,
) -> Dict[str, float] | Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    graph_tensors = None
    if graph_info:
        graph_tensors = {
            'adj': torch.from_numpy(graph_info['adj']).float().to(device),
            'edge_index': torch.from_numpy(graph_info['edge_index']).long().to(device),
        }

    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if visible_steps is not None:
            x = _mask_future_steps(x, visible_steps)

        autocast_ctx = (
            torch.autocast(device_type='cuda', dtype=torch.float16)
            if amp_enabled and device.type == 'cuda'
            else nullcontext()
        )
        with autocast_ctx:
            pred = _forward_model(model, x, model_type, graph_tensors)
            loss, _ = _compute_loss(pred, y, model, criterion, model_type)

        total_loss += float(loss.item())
        all_preds.append(pred.cpu())
        all_labels.append(y.cpu())

    all_preds_tensor = torch.cat(all_preds, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_preds_tensor, all_labels_tensor, threshold=threshold)
    metrics['loss'] = total_loss / max(1, len(val_loader))

    if return_raw:
        return metrics, all_preds_tensor.numpy(), all_labels_tensor.numpy()
    return metrics


def evaluate_online_prefix(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_type: str,
    graph_info: Optional[Dict[str, np.ndarray]],
    threshold: float,
    amp_enabled: bool,
    seq_len: int,
    prefix_fractions: List[float],
    min_visible_steps: int,
) -> Dict[str, float]:
    step_metrics: List[Dict[str, float]] = []
    for fraction in prefix_fractions:
        visible_steps = max(int(round(seq_len * float(fraction))), int(min_visible_steps))
        metrics = evaluate(
            model,
            data_loader,
            criterion,
            device,
            model_type=model_type,
            graph_info=graph_info,
            threshold=threshold,
            return_raw=False,
            amp_enabled=amp_enabled,
            visible_steps=visible_steps,
        )
        step_metrics.append(metrics)

    return {
        'prefix_mae_mean': float(np.mean([m['mae'] for m in step_metrics])),
        'prefix_rmse_mean': float(np.mean([m['rmse'] for m in step_metrics])),
        'prefix_f1_mean': float(np.mean([m['f1'] for m in step_metrics])),
        'prefix_iou_mean': float(np.mean([m['iou'] for m in step_metrics])),
        'prefix_eval_count': float(len(step_metrics)),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    model_type: str = 'gt',
    graph_info: Optional[Dict[str, np.ndarray]] = None,
    checkpoint_path: str = 'checkpoints/model_best.pth',
    checkpoint_payload_base: Optional[Dict[str, Any]] = None,
    threshold: float = 0.95,
    amp_enabled: bool = False,
    task_mode: str = 'final_state',
    online_prefix_min_visible: int = 1,
) -> Dict[str, List[float]]:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    if model_type == 'legacy_pinn':
        criterion = PINNLoss(
            lambda_data=1.0,
            lambda_smooth=config.get('lambda_smooth', 0.1),
            lambda_range=0.01,
            lambda_magnitude=0.05,
        )
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    early_stopping = EarlyStopping(patience=config.get('patience', 20))
    scaler = _build_grad_scaler(amp_enabled=amp_enabled, device=device)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_f1': [],
    }

    best_val_loss = float('inf')
    for epoch in range(config.get('epochs', 100)):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            model_type=model_type,
            graph_info=graph_info,
            amp_enabled=amp_enabled,
            scaler=scaler,
            task_mode=task_mode,
            online_prefix_min_visible=online_prefix_min_visible,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            model_type=model_type,
            graph_info=graph_info,
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
            payload = dict(checkpoint_payload_base or {})
            payload['model_state_dict'] = model.state_dict()
            payload['epoch'] = int(epoch)
            payload['best_val_loss'] = float(best_val_loss)
            torch.save(payload, checkpoint_path)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}: train_loss={train_metrics['loss']:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, val_mae={val_metrics['mae']:.4f}, "
                f"val_f1={val_metrics['f1']:.4f}"
            )

        if early_stopping(val_metrics['loss']):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return history


def _choose_threshold(preds: np.ndarray, labels: np.ndarray, fixed_threshold: float, strategy: str) -> float:
    if strategy == 'fixed':
        return float(fixed_threshold)

    candidate_thresholds = np.linspace(0.55, 0.99, 45)
    best_threshold = float(fixed_threshold)
    best_score = -np.inf

    pred_tensor = torch.from_numpy(preds)
    label_tensor = torch.from_numpy(labels)
    for threshold in candidate_thresholds:
        metrics = compute_metrics(pred_tensor, label_tensor, threshold=float(threshold))
        score = metrics['f1'] + 0.5 * metrics['balanced_accuracy']
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _make_results_serializable(results: Dict[str, Any]) -> Dict[str, Any]:
    serializable_results: Dict[str, Any] = {}
    for model_name, model_results in results.items():
        serializable_results[model_name] = {
            'history': {k: [float(v) for v in vals] for k, vals in model_results['history'].items()},
            'test_metrics': {k: float(v) for k, v in model_results['test_metrics'].items()},
        }
        online_metrics = model_results.get('online_test_metrics')
        if online_metrics:
            serializable_results[model_name]['online_test_metrics'] = {
                k: float(v) for k, v in online_metrics.items()
            }
    return serializable_results


def _build_gt_model(
    checkpoint_payload: Optional[Dict[str, Any]],
    num_nodes: int,
    num_elements: int,
    seq_len: int,
    node_in_dim: int,
    hidden_dim: int,
) -> Tuple[nn.Module, Dict[str, Any]]:
    if checkpoint_payload and checkpoint_payload.get('model_args'):
        model_args = dict(checkpoint_payload['model_args'])
    elif checkpoint_payload and 'model_state_dict' in checkpoint_payload:
        state_dict = checkpoint_payload['model_state_dict']
        model_args = {
            'num_nodes': num_nodes,
            'num_elements': num_elements,
            'seq_len': seq_len,
            'node_in_dim': _infer_gt_node_in_dim(state_dict),
            'hidden_dim': _infer_gt_hidden_dim(state_dict),
        }
    else:
        model_args = {
            'num_nodes': num_nodes,
            'num_elements': num_elements,
            'seq_len': seq_len,
            'node_in_dim': node_in_dim,
            'hidden_dim': hidden_dim,
        }
    return GTDamagePredictor(**model_args), model_args


def _build_legacy_pinn_model(
    checkpoint_payload: Optional[Dict[str, Any]],
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
) -> Tuple[nn.Module, Dict[str, Any]]:
    if checkpoint_payload and checkpoint_payload.get('model_args'):
        model_args = dict(checkpoint_payload['model_args'])
    elif checkpoint_payload and 'model_state_dict' in checkpoint_payload:
        state_dict = checkpoint_payload['model_state_dict']
        model_args = {
            'input_dim': _infer_legacy_pinn_input_dim(state_dict),
            'hidden_dim': _infer_legacy_pinn_hidden_dim(state_dict),
            'num_layers': _infer_legacy_pinn_num_layers(state_dict),
            'output_dim': output_dim,
        }
    else:
        model_args = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
        }
    return PINNDamagePredictor(**model_args), model_args


def _build_pinn_v2_model(
    checkpoint_payload: Optional[Dict[str, Any]],
    input_dim: int,
    output_dim: int,
    train_cfg: Dict[str, Any],
) -> Tuple[nn.Module, Dict[str, Any]]:
    if checkpoint_payload and checkpoint_payload.get('model_args'):
        model_args = dict(checkpoint_payload['model_args'])
    else:
        model_args = {
            'input_dim': input_dim,
            'hidden_dims': train_cfg.get('hidden_dims', [256, 256, 128]),
            'output_dim': output_dim,
            'num_residual_blocks': int(train_cfg.get('num_residual_blocks', 2)),
            'dropout': float(train_cfg.get('dropout', 0.3)),
            'use_physics': bool(train_cfg.get('use_physics', True)),
        }
    return PINNDamagePredictorV2(**model_args), model_args


def main() -> None:
    parser = argparse.ArgumentParser(description='Train damage detection models')
    parser.add_argument('--config', type=str, default=os.path.join(ROOT_DIR, 'dataset_config.yaml'))
    parser.add_argument('--model', type=str, default=None, choices=['gt', 'pinn', 'legacy_pinn', 'pinn_v2', 'both'])
    parser.add_argument('--dataset_mode', type=str, default=None, choices=['response', 'load', 'both'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--structure_file', type=str, default=None)
    parser.add_argument('--lambda_smooth', type=float, default=None)
    parser.add_argument('--threshold', type=float, default=None, help='Fixed damage classification threshold')
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
    pinn_v2_cfg = _load_train_config(os.path.join(SCRIPT_DIR, 'configs', 'pinn_v2.yaml')).get('train', {})

    def pick(cli_value, cfg_key: str, default_value):
        if cli_value is not None:
            return cli_value
        return train_cfg.get(cfg_key, default_value)

    model_choice = _normalize_model_choice(str(pick(args.model, 'model', 'gt')))
    dataset_mode = str(pick(args.dataset_mode, 'dataset_mode', 'response')).lower()
    epochs = int(pick(args.epochs, 'epochs', 100))
    batch_size = int(pick(args.batch_size, 'batch_size', 32))
    lr = float(pick(args.lr, 'lr', 1e-3))
    hidden_dim = int(pick(args.hidden_dim, 'hidden_dim', 64))
    lambda_smooth = float(pick(args.lambda_smooth, 'lambda_smooth', 0.1))
    fixed_threshold = float(pick(args.threshold, 'threshold', 0.95))
    threshold_strategy = str(train_cfg.get('threshold_strategy', 'val_optimal' if args.threshold is None else 'fixed')).lower()
    seed = int(pick(args.seed, 'seed', 42))
    patience = int(pick(args.patience, 'patience', 20))
    deterministic = bool(pick(args.deterministic, 'deterministic', True))
    deterministic_warn_only = bool(pick(args.deterministic_warn_only, 'deterministic_warn_only', False))
    num_threads = int(pick(args.num_threads, 'num_threads', 1))
    device_request = str(pick(args.device, 'device', 'auto'))
    amp_enabled = bool(pick(args.amp, 'amp', False))
    num_workers = int(pick(args.num_workers, 'num_workers', 0))
    pin_memory_cfg = bool(pick(args.pin_memory, 'pin_memory', True))
    split_strategy = str(train_cfg.get('split_strategy', 'random')).lower()
    damage_holdout_threshold = train_cfg.get('damage_holdout_threshold')
    task_mode = str(train_cfg.get('task_mode', 'final_state')).lower()
    online_prefix_min_visible = int(train_cfg.get('online_prefix_min_visible', 1))
    online_eval_prefix_fractions = [float(x) for x in train_cfg.get('online_eval_prefix_fractions', [0.25, 0.5, 0.75, 1.0])]

    data_path_cfg = args.data_path or cfg.get('output_file', 'dataset/train.npz')
    data_path = _resolve_path(cfg_dir, data_path_cfg)
    structure_path_cfg = args.structure_file or cfg.get('structure_file', 'structure.yaml')
    structure_file = _resolve_path(cfg_dir, structure_path_cfg)
    ckpt_dir = os.path.join(SCRIPT_DIR, 'checkpoints') if args.output_dir is None else _resolve_path(SCRIPT_DIR, args.output_dir)
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
    print(f"Config: {args.config}")
    print(f"Loading data from {data_path}")
    print(f"Structure file: {structure_file}")
    print(f"Task mode: {task_mode}")
    print(f"Split strategy: {split_strategy}")
    print(f"Threshold strategy: {threshold_strategy}")

    dataset = FEMDataset(data_path, mode=dataset_mode, normalize=True)
    train_ratio = float(train_cfg.get('train_ratio', 0.7))
    val_ratio = float(train_cfg.get('val_ratio', 0.15))
    test_ratio = float(train_cfg.get('test_ratio', 0.15))
    train_set, val_set, test_set = split_dataset(
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        strategy=split_strategy,
        damage_holdout_threshold=damage_holdout_threshold,
    )
    dataset.fit_normalization(train_set.indices)
    preprocessing_stats = dataset.get_normalization_stats()

    train_loader_generator = torch.Generator().manual_seed(seed)
    worker_init_fn = partial(_seed_worker, base_seed=seed) if num_workers > 0 else None
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=train_loader_generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(num_workers > 0),
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
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
        raise ValueError(f"Dataset output_dim={output_dim} does not match structure num_elements={num_elements}")
    if input_dim % num_nodes != 0:
        raise ValueError(f"Input dim {input_dim} is not divisible by num_nodes {num_nodes}.")

    node_in_dim = input_dim // num_nodes
    adj, edge_index = build_graph_info(element_conn, num_nodes=num_nodes)
    graph_info = {'adj': adj, 'edge_index': edge_index}
    print(f"Sequence length: {seq_len}, Input dim: {input_dim}, Node in dim: {node_in_dim}, Output dim: {output_dim}")
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    base_metadata = {
        'seed': int(seed),
        'config_path': os.path.abspath(args.config),
        'data_path': os.path.abspath(data_path),
        'structure_file': os.path.abspath(structure_file),
        'dataset_mode': dataset_mode,
        'task_mode': task_mode,
        'split_strategy': split_strategy,
        'damage_holdout_threshold': float(damage_holdout_threshold) if damage_holdout_threshold is not None else None,
    }

    base_config = {
        'epochs': epochs,
        'lr': lr,
        'patience': patience,
        'lambda_smooth': lambda_smooth,
    }

    results: Dict[str, Any] = {}
    prediction_artifacts: Dict[str, np.ndarray] = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_models = [model_choice] if model_choice != 'both' else ['gt', 'pinn_v2']
    for run_model in run_models:
        run_model = _normalize_model_choice(run_model)
        if run_model == 'gt':
            checkpoint_path = os.path.join(ckpt_dir, 'gt_best.pth')
            checkpoint_payload = _load_checkpoint_payload(checkpoint_path, map_location=device) if (args.eval_only and os.path.exists(checkpoint_path)) else None
            model, model_args = _build_gt_model(checkpoint_payload, num_nodes, output_dim, seq_len, node_in_dim, hidden_dim)
            model = model.to(device)
            checkpoint_payload_base = _make_checkpoint_payload(
                model,
                model_name='gt',
                model_args=model_args,
                preprocessing=preprocessing_stats,
                training_task=task_mode,
                feature_mode=dataset_mode,
                extra_metadata=dict(base_metadata),
            )
            history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_f1': []}
            if not args.eval_only:
                history = train_model(
                    model,
                    train_loader,
                    val_loader,
                    base_config,
                    device,
                    model_type='gt',
                    graph_info=graph_info,
                    checkpoint_path=checkpoint_path,
                    checkpoint_payload_base=checkpoint_payload_base,
                    threshold=fixed_threshold,
                    amp_enabled=amp_enabled,
                    task_mode=task_mode,
                    online_prefix_min_visible=online_prefix_min_visible,
                )
                checkpoint_payload = _load_checkpoint_payload(checkpoint_path, map_location=device)
            elif checkpoint_payload is None:
                raise FileNotFoundError(f"Checkpoint not found for eval_only mode: {checkpoint_path}")

            model.load_state_dict(checkpoint_payload['model_state_dict'])
            _, val_pred, val_true = evaluate(
                model,
                val_loader,
                nn.MSELoss(),
                device,
                model_type='gt',
                graph_info=graph_info,
                threshold=fixed_threshold,
                return_raw=True,
                amp_enabled=amp_enabled,
            )
            selected_threshold = _choose_threshold(val_pred, val_true, fixed_threshold, threshold_strategy)
            checkpoint_payload['selected_threshold'] = float(selected_threshold)
            torch.save(checkpoint_payload, checkpoint_path)

            test_metrics, test_pred, test_true = evaluate(
                model,
                test_loader,
                nn.MSELoss(),
                device,
                model_type='gt',
                graph_info=graph_info,
                threshold=selected_threshold,
                return_raw=True,
                amp_enabled=amp_enabled,
            )
            online_test_metrics = evaluate_online_prefix(
                model,
                test_loader,
                nn.MSELoss(),
                device,
                model_type='gt',
                graph_info=graph_info,
                threshold=selected_threshold,
                amp_enabled=amp_enabled,
                seq_len=seq_len,
                prefix_fractions=online_eval_prefix_fractions,
                min_visible_steps=online_prefix_min_visible,
            )
            prediction_artifacts['gt_pred'] = test_pred.astype(np.float32)
            prediction_artifacts['true_damage'] = test_true.astype(np.float32)
            results['gt'] = {
                'history': history,
                'test_metrics': test_metrics,
                'online_test_metrics': online_test_metrics,
            }
            print(f"\nGT Test Results: MAE={test_metrics['mae']:.4f}, F1={test_metrics['f1']:.4f}, threshold={selected_threshold:.3f}")

        elif run_model == 'legacy_pinn':
            checkpoint_path = os.path.join(ckpt_dir, 'pinn_best.pth')
            checkpoint_payload = _load_checkpoint_payload(checkpoint_path, map_location=device) if (args.eval_only and os.path.exists(checkpoint_path)) else None
            model, model_args = _build_legacy_pinn_model(checkpoint_payload, input_dim, output_dim, hidden_dim)
            model = model.to(device)
            adjacency = build_element_adjacency(element_conn)
            model.set_structure_info(adjacency)
            checkpoint_payload_base = _make_checkpoint_payload(
                model,
                model_name='legacy_pinn',
                model_args=model_args,
                preprocessing=preprocessing_stats,
                training_task=task_mode,
                feature_mode=dataset_mode,
                extra_metadata=dict(base_metadata),
            )
            history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_f1': []}
            if not args.eval_only:
                history = train_model(
                    model,
                    train_loader,
                    val_loader,
                    base_config,
                    device,
                    model_type='legacy_pinn',
                    graph_info=None,
                    checkpoint_path=checkpoint_path,
                    checkpoint_payload_base=checkpoint_payload_base,
                    threshold=fixed_threshold,
                    amp_enabled=amp_enabled,
                    task_mode=task_mode,
                    online_prefix_min_visible=online_prefix_min_visible,
                )
                checkpoint_payload = _load_checkpoint_payload(checkpoint_path, map_location=device)
            elif checkpoint_payload is None:
                raise FileNotFoundError(f"Checkpoint not found for eval_only mode: {checkpoint_path}")

            model.load_state_dict(checkpoint_payload['model_state_dict'])
            criterion = PINNLoss(
                lambda_data=1.0,
                lambda_smooth=base_config.get('lambda_smooth', 0.1),
                lambda_range=0.01,
                lambda_magnitude=0.05,
            )
            _, val_pred, val_true = evaluate(
                model,
                val_loader,
                criterion,
                device,
                model_type='legacy_pinn',
                graph_info=None,
                threshold=fixed_threshold,
                return_raw=True,
                amp_enabled=amp_enabled,
            )
            selected_threshold = _choose_threshold(val_pred, val_true, fixed_threshold, threshold_strategy)
            checkpoint_payload['selected_threshold'] = float(selected_threshold)
            torch.save(checkpoint_payload, checkpoint_path)

            test_metrics, test_pred, test_true = evaluate(
                model,
                test_loader,
                criterion,
                device,
                model_type='legacy_pinn',
                graph_info=None,
                threshold=selected_threshold,
                return_raw=True,
                amp_enabled=amp_enabled,
            )
            online_test_metrics = evaluate_online_prefix(
                model,
                test_loader,
                criterion,
                device,
                model_type='legacy_pinn',
                graph_info=None,
                threshold=selected_threshold,
                amp_enabled=amp_enabled,
                seq_len=seq_len,
                prefix_fractions=online_eval_prefix_fractions,
                min_visible_steps=online_prefix_min_visible,
            )
            prediction_artifacts['legacy_pinn_pred'] = test_pred.astype(np.float32)
            if 'true_damage' not in prediction_artifacts:
                prediction_artifacts['true_damage'] = test_true.astype(np.float32)
            results['legacy_pinn'] = {
                'history': history,
                'test_metrics': test_metrics,
                'online_test_metrics': online_test_metrics,
            }
            print(
                f"\nLegacy PINN Test Results: MAE={test_metrics['mae']:.4f}, "
                f"F1={test_metrics['f1']:.4f}, threshold={selected_threshold:.3f}"
            )

        elif run_model == 'pinn_v2':
            checkpoint_path = os.path.join(ckpt_dir, 'pinn_v2_best.pth')
            checkpoint_payload = _load_checkpoint_payload(checkpoint_path, map_location=device) if (args.eval_only and os.path.exists(checkpoint_path)) else None
            model, model_args = _build_pinn_v2_model(checkpoint_payload, input_dim, output_dim, pinn_v2_cfg)
            model = model.to(device)
            adjacency = build_element_adjacency(element_conn)
            model.set_structure_info(element_adjacency=adjacency)
            train_cfg_v2 = dict(pinn_v2_cfg)
            train_cfg_v2.update({
                'epochs': epochs,
                'lr': lr,
                'patience': patience,
            })
            history = {
                'train_loss': [],
                'train_data_loss': [],
                'train_physics_loss': [],
                'val_loss': [],
                'val_mae': [],
                'val_rmse': [],
                'val_f1': [],
                'data_weight': [],
                'physics_weight': [],
            }
            if not args.eval_only:
                model, history = train_pinn_v2_model(
                    model,
                    train_loader,
                    val_loader,
                    train_cfg_v2,
                    device,
                    save_dir=ckpt_dir,
                    verbose=True,
                )
                checkpoint_payload = _load_checkpoint_payload(checkpoint_path, map_location=device)
                checkpoint_payload.update(
                    _make_checkpoint_payload(
                        model,
                        model_name='pinn_v2',
                        model_args=model_args,
                        preprocessing=preprocessing_stats,
                        training_task=task_mode,
                        feature_mode=dataset_mode,
                        extra_metadata=dict(base_metadata),
                    )
                )
                torch.save(checkpoint_payload, checkpoint_path)
            elif checkpoint_payload is None:
                raise FileNotFoundError(f"Checkpoint not found for eval_only mode: {checkpoint_path}")

            model.load_state_dict(checkpoint_payload['model_state_dict'])
            raw_val_metrics, val_pred, val_true = evaluate_pinn_v2_loader(model, val_loader, device, return_raw=True)
            selected_threshold = _choose_threshold(val_pred, val_true, fixed_threshold, threshold_strategy)
            checkpoint_payload['selected_threshold'] = float(selected_threshold)
            torch.save(checkpoint_payload, checkpoint_path)

            raw_test_metrics, test_pred, test_true = evaluate_pinn_v2_loader(model, test_loader, device, return_raw=True)
            test_metrics = compute_metrics(torch.from_numpy(test_pred), torch.from_numpy(test_true), threshold=selected_threshold)
            test_metrics['loss'] = float(raw_test_metrics['loss'])
            online_test_metrics = evaluate_online_prefix(
                model,
                test_loader,
                nn.MSELoss(),
                device,
                model_type='pinn_v2',
                graph_info=None,
                threshold=selected_threshold,
                amp_enabled=False,
                seq_len=seq_len,
                prefix_fractions=online_eval_prefix_fractions,
                min_visible_steps=online_prefix_min_visible,
            )
            prediction_artifacts['pinn_v2_pred'] = test_pred.astype(np.float32)
            if 'true_damage' not in prediction_artifacts:
                prediction_artifacts['true_damage'] = test_true.astype(np.float32)
            results['pinn_v2'] = {
                'history': history,
                'test_metrics': test_metrics,
                'online_test_metrics': online_test_metrics,
            }
            print(f"\nPINN V2 Test Results: MAE={test_metrics['mae']:.4f}, F1={test_metrics['f1']:.4f}, threshold={selected_threshold:.3f}")

    if model_choice == 'both':
        print("\n" + "=" * 50)
        print("Model Comparison")
        print("=" * 50)
        print(f"{'Metric':<15} {'GT':<15} {'PINN_V2':<15}")
        print("-" * 45)
        for metric in ['mae', 'rmse', 'f1', 'iou']:
            gt_val = results['gt']['test_metrics'].get(metric, 0)
            pinn_val = results['pinn_v2']['test_metrics'].get(metric, 0)
            print(f"{metric:<15} {gt_val:<15.4f} {pinn_val:<15.4f}")

    results_file = os.path.join(ckpt_dir, f'results_{timestamp}.json')
    prediction_file: Optional[str] = None
    if prediction_artifacts:
        prediction_file = os.path.join(ckpt_dir, f'predictions_{timestamp}.npz')
        np.savez_compressed(prediction_file, **prediction_artifacts)

    output_payload = {
        'metadata': {
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'fixed_threshold': float(fixed_threshold),
            'threshold_strategy': threshold_strategy,
            'seed': int(seed),
            'device': str(device),
            'config_path': os.path.abspath(args.config),
            'data_path': os.path.abspath(data_path),
            'structure_file': os.path.abspath(structure_file),
            'dataset_mode': dataset_mode,
            'task_mode': task_mode,
            'split_strategy': split_strategy,
            'deterministic': deterministic,
            'deterministic_warn_only': deterministic_warn_only,
            'num_threads': int(num_threads),
            'device_request': device_request,
            'amp_enabled': bool(amp_enabled),
            'num_workers': int(num_workers),
            'pin_memory': bool(pin_memory),
            'eval_only': bool(args.eval_only),
            'prediction_file': prediction_file,
        },
        'models': _make_results_serializable(results),
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {results_file}")
    if prediction_file:
        print(f"Predictions saved to {prediction_file}")


if __name__ == '__main__':
    main()
