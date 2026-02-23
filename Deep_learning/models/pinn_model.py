
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any


class PINNDamagePredictor(nn.Module):
    """
    方案三: 物理嵌入神经网络 (PINN)
    
    结合物理约束的损伤识别模型。
    
    损失函数 = Data Loss + λ1 * Equilibrium Loss + λ2 * Smoothness Loss
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        output_dim: int = 11,
        dropout: float = 0.3
    ):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: 隐藏层数量
            output_dim: 输出维度 (单元数量)
            dropout: Dropout概率
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.encoder = nn.Sequential(*layers)
        
        self.damage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
        
        self.feature_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def set_structure_info(
        self,
        element_adjacency: np.ndarray,
        stiffness_importance: Optional[np.ndarray] = None
    ):
        """
        设置结构拓扑信息用于物理约束。
        
        Args:
            element_adjacency: 单元邻接矩阵 (num_elem, num_elem)
            stiffness_importance: 各单元刚度重要性权重
        """
        self.register_buffer(
            'element_adjacency',
            torch.from_numpy(element_adjacency).float()
        )
        
        if stiffness_importance is not None:
            self.register_buffer(
                'stiffness_importance',
                torch.from_numpy(stiffness_importance).float()
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, timesteps, features) 或 (batch, features)
        
        Returns:
            dict with keys:
                'damage': (batch, num_elements) 损伤程度
                'feature': (batch, num_elements) 特征表示
        """
        if x.dim() == 3:
            x = x.mean(dim=1)
        
        encoded = self.encoder(x)
        
        damage = self.damage_head(encoded)
        feature = self.feature_head(encoded)
        
        return {
            'damage': damage,
            'feature': feature
        }
    
    def compute_physics_loss(
        self,
        damage: torch.Tensor,
        lambda_smooth: float = 0.1,
        lambda_range: float = 0.01
    ) -> Dict[str, torch.Tensor]:
        """
        计算物理约束损失。
        
        Args:
            damage: 预测的损伤程度 (batch, num_elements)
            lambda_smooth: 平滑性损失权重
            lambda_range: 范围约束损失权重
        
        Returns:
            dict of loss components
        """
        losses = {}
        
        if self.element_adjacency is not None:
            adj = self.element_adjacency
            diff = damage.unsqueeze(-1) - damage.unsqueeze(-2)
            smooth_loss = (adj * diff ** 2).mean()
            losses['smoothness_loss'] = smooth_loss
        
        range_loss = torch.mean(torch.relu(damage - 1.0) ** 2 + torch.relu(-damage) ** 2)
        losses['range_loss'] = range_loss
        
        total_loss = lambda_smooth * losses.get('smoothness_loss', 0) + lambda_range * range_loss
        losses['total_physics_loss'] = total_loss
        
        return losses


class PINNLoss(nn.Module):
    """
    PINN组合损失函数
    """
    
    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_smooth: float = 0.1,
        lambda_range: float = 0.01,
        lambda_magnitude: float = 0.05
    ):
        super().__init__()
        
        self.lambda_data = lambda_data
        self.lambda_smooth = lambda_smooth
        self.lambda_range = lambda_range
        self.lambda_magnitude = lambda_magnitude
        
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
    
    def forward(
        self,
        pred_damage: torch.Tensor,
        true_damage: torch.Tensor,
        physics_losses: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算总损失。
        
        Args:
            pred_damage: 预测损伤 (batch, num_elements)
            true_damage: 真实损伤 (batch, num_elements)
            physics_losses: 物理损失字典
        
        Returns:
            损失字典
        """
        losses = {}
        
        data_loss = self.mse_loss(pred_damage, true_damage)
        losses['data_loss'] = data_loss
        
        damage_mask = (true_damage < 0.95).float()
        classification_loss = self.bce_loss(
            (1 - pred_damage) * damage_mask + pred_damage * (1 - damage_mask),
            (true_damage < 0.95).float()
        )
        losses['classification_loss'] = classification_loss
        
        damage_deviation = torch.abs(pred_damage - true_damage)
        magnitude_loss = (damage_deviation * (true_damage < 0.95).float()).mean()
        losses['magnitude_loss'] = magnitude_loss
        
        total_loss = (
            self.lambda_data * data_loss +
            0.5 * classification_loss +
            self.lambda_magnitude * magnitude_loss
        )
        
        if physics_losses is not None:
            total_loss = total_loss + physics_losses.get('total_physics_loss', 0)
            losses.update(physics_losses)
        
        losses['total_loss'] = total_loss
        
        return losses
