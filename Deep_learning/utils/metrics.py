
import torch
import numpy as np
from typing import Dict, Optional


def compute_metrics(
    pred: torch.Tensor,
    true: torch.Tensor,
    threshold: float = 0.95
) -> Dict[str, float]:
    """
    计算评估指标。
    
    Args:
        pred: 预测值 (batch, num_elements)
        true: 真实值 (batch, num_elements)
        threshold: 损伤判定阈值
    
    Returns:
        指标字典
    """
    pred = pred.detach().cpu().numpy()
    true = true.detach().cpu().numpy()
    
    mse = np.mean((pred - true) ** 2)
    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(mse)
    
    pred_binary = (pred < threshold).astype(int)
    true_binary = (true < threshold).astype(int)
    
    tp = np.sum((pred_binary == 1) & (true_binary == 1))
    fp = np.sum((pred_binary == 1) & (true_binary == 0))
    fn = np.sum((pred_binary == 0) & (true_binary == 1))
    tn = np.sum((pred_binary == 0) & (true_binary == 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    balanced_accuracy = 0.5 * (recall + specificity)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    iou = tp / (tp + fp + fn + 1e-8)
    
    damaged_mask = true < threshold
    if damaged_mask.sum() > 0:
        damage_mae = np.mean(np.abs(pred[damaged_mask] - true[damaged_mask]))
    else:
        damage_mae = 0.0
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'balanced_accuracy': float(balanced_accuracy),
        'f1': float(f1),
        'iou': float(iou),
        'damage_mae': float(damage_mae),
        'threshold': float(threshold),
        'support_damaged': int(np.sum(true_binary)),
        'support_undamaged': int(np.sum(1 - true_binary))
    }


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: 容忍轮数
            min_delta: 最小改进阈值
            mode: 'min' 或 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
