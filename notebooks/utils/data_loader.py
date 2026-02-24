import numpy as np
import yaml
import json
from typing import Dict, Any, Tuple, Optional


def load_dataset(npz_path: str, normalize: bool = True) -> Dict[str, np.ndarray]:
    """加载 npz 数据集
    
    Args:
        npz_path: 数据集文件路径
        normalize: 是否返回归一化后的数据
        
    Returns:
        包含 load, disp, stress, damage 的字典
    """
    if not npz_path.endswith('.npz'):
        npz_path = npz_path + '.npz'
    
    data = np.load(npz_path)
    
    result = {
        'load': data['load'],
        'disp': data['disp'],
        'stress': data['stress'],
        'damage': data['damage'],
    }
    
    if normalize:
        load_mean = np.mean(result['load'])
        load_std = np.std(result['load'])
        disp_mean = np.mean(result['disp'])
        disp_std = np.std(result['disp'])
        stress_mean = np.mean(result['stress'])
        stress_std = np.std(result['stress'])
        
        result['load_norm'] = (result['load'] - load_mean) / (load_std + 1e-8)
        result['disp_norm'] = (result['disp'] - disp_mean) / (disp_std + 1e-8)
        result['stress_norm'] = (result['stress'] - stress_mean) / (stress_std + 1e-8)
        result['_norm_params'] = {
            'load_mean': load_mean, 'load_std': load_std,
            'disp_mean': disp_mean, 'disp_std': disp_std,
            'stress_mean': stress_mean, 'stress_std': stress_std,
        }
    
    return result


def load_structure_config(yaml_path: str) -> Dict[str, Any]:
    """加载结构配置文件
    
    Args:
        yaml_path: 结构配置文件路径
        
    Returns:
        包含结构配置的字典
    """
    if not yaml_path.endswith('.yaml'):
        yaml_path = yaml_path + '.yaml'
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_training_results(json_path: str) -> Dict[str, Any]:
    """加载训练结果 JSON
    
    Args:
        json_path: 训练结果文件路径
        
    Returns:
        包含训练历史的字典
    """
    if not json_path.endswith('.json'):
        json_path = json_path + '.json'
    
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    return results


def load_predictions(npz_path: str, model: str = 'gt') -> Tuple[np.ndarray, np.ndarray]:
    """加载预测结果
    
    Args:
        npz_path: 预测结果文件路径
        model: 模型名称 ('gt' 或 'pinn')
        
    Returns:
        (y_true, y_pred) 元组
    """
    if not npz_path.endswith('.npz'):
        npz_path = npz_path + '.npz'
    
    data = np.load(npz_path)
    keys = list(data.keys())
    
    if 'y_true' in data and 'y_pred' in data:
        return data['y_true'], data['y_pred']
    elif 'damage_true' in data and 'damage_pred' in data:
        return data['damage_true'], data['damage_pred']
    elif 'true_damage' in data:
        y_true = data['true_damage']
        if model == 'gt' and 'gt_pred' in data:
            y_pred = data['gt_pred']
        elif model == 'pinn' and 'pinn_pred' in data:
            y_pred = data['pinn_pred']
        elif 'y_pred' in data:
            y_pred = data['y_pred']
        else:
            raise ValueError(f"Unknown model '{model}' in npz file. Available keys: {keys}")
        return y_true, y_pred
    else:
        raise ValueError(f"Unknown keys in npz file: {keys}. Expected 'y_true'/'y_pred' or 'true_damage'/'gt_pred'/'pinn_pred'")


def get_time_array(num_steps: int, dt: float) -> np.ndarray:
    """生成时间数组
    
    Args:
        num_steps: 时间步数
        dt: 时间步长
        
    Returns:
        时间数组
    """
    return np.linspace(0, dt * (num_steps - 1), num_steps)


def get_latest_results() -> Tuple[Optional[str], Optional[str]]:
    """获取最新的结果和预测文件路径
    
    Returns:
        (results_json_path, predictions_npz_path) 或 (None, None) 如果不存在
    """
    import os
    from .config import CHECKPOINT_DIR
    
    results_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('results_') and f.endswith('.json')]
    pred_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('predictions_') and f.endswith('.npz')]
    
    if not results_files or not pred_files:
        return None, None
    
    results_files.sort(reverse=True)
    pred_files.sort(reverse=True)
    
    return os.path.join(CHECKPOINT_DIR, results_files[0]), os.path.join(CHECKPOINT_DIR, pred_files[0])


def load_checkpoint_model(model_path: str, model_class=None, device: str = 'cpu'):
    """加载 PyTorch 模型
    
    Args:
        model_path: 模型权重文件路径
        model_class: 模型类（需要用户提供）
        device: 设备类型
        
    Returns:
        加载好的模型
    """
    import torch
    
    if model_class is None:
        raise ValueError("model_class must be provided to load the checkpoint")
    
    model = model_class()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model
