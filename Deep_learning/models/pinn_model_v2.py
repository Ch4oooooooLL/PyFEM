"""
Optimized PINN Damage Predictor V2.

Improvements over V1:
- Better weight initialization (Kaiming)
- Residual connections for gradient flow
- Integrated adaptive physics weighting
- Layer normalization for stability
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PINNDamagePredictorV2(nn.Module):
    """
    优化版物理嵌入神经网络 (PINN V2)
    
    Improvements:
    - Kaiming weight initialization
    - Residual connections for deeper networks
    - Layer normalization for training stability
    - Integrated adaptive physics constraints
    - Multi-scale feature extraction
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        output_dim: int = 11,
        num_residual_blocks: int = 2,
        dropout: float = 0.3,
        use_physics: bool = True,
        physics_weight: float = 0.1
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (number of elements)
            num_residual_blocks: Number of residual blocks
            dropout: Dropout probability
            use_physics: Whether to use physics constraints
            physics_weight: Initial physics loss weight
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_physics = use_physics
        self.physics_weight = physics_weight
        
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]
        
        # Build encoder with residual connections
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer with Kaiming initialization
            linear = nn.Linear(prev_dim, hidden_dim)
            nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(linear.bias)
            
            layers.append(linear)
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            # Add residual blocks after each layer (except last)
            if i < len(hidden_dims) - 1:
                for _ in range(num_residual_blocks):
                    layers.append(ResidualBlock(hidden_dim, dropout))
            
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.encoder_dim = hidden_dims[-1]
        
        # Multi-scale feature extraction
        self.feature_pyramid = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.encoder_dim, self.encoder_dim // 2),
                nn.LayerNorm(self.encoder_dim // 2),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Linear(self.encoder_dim // 2, self.encoder_dim // 4),
                nn.LayerNorm(self.encoder_dim // 4),
                nn.ReLU()
            )
        ])
        
        # Damage prediction head
        self.damage_head = nn.Sequential(
            nn.Linear(self.encoder_dim // 4 + self.encoder_dim // 2, self.encoder_dim // 4),
            nn.LayerNorm(self.encoder_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.encoder_dim // 4, output_dim)
        )
        
        # Initialize damage head
        nn.init.xavier_uniform_(self.damage_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.damage_head[-1].bias)
        
        # Physics feature head (for computing physics constraints)
        self.physics_head = nn.Sequential(
            nn.Linear(self.encoder_dim // 4, self.encoder_dim // 8),
            nn.LayerNorm(self.encoder_dim // 8),
            nn.ReLU(),
            nn.Linear(self.encoder_dim // 8, output_dim)
        )
        
        # Sigmoid activation for damage output
        self.sigmoid = nn.Sigmoid()
        
        # Structure info (set externally)
        self.element_adjacency = None
        self.stiffness_importance = None
        self.element_coords = None
        
    def set_structure_info(
        self,
        element_adjacency: Optional[np.ndarray] = None,
        stiffness_importance: Optional[np.ndarray] = None,
        element_coords: Optional[np.ndarray] = None
    ):
        """
        Set structure topology information for physics constraints.
        
        Args:
            element_adjacency: Element adjacency matrix (num_elem, num_elem)
            stiffness_importance: Element stiffness importance weights
            element_coords: Element coordinates for spatial constraints (num_elem, 2)
        """
        if element_adjacency is not None:
            self.register_buffer(
                'element_adjacency',
                torch.from_numpy(element_adjacency).float()
            )
        
        if stiffness_importance is not None:
            self.register_buffer(
                'stiffness_importance',
                torch.from_numpy(stiffness_importance).float()
            )
        
        if element_coords is not None:
            self.register_buffer(
                'element_coords',
                torch.from_numpy(element_coords).float()
            )
    
    def forward(
        self,
        x: torch.Tensor,
        E: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (batch, timesteps, features) or (batch, features)
            E: Young's modulus (batch,) - optional, for physics constraints
        
        Returns:
            damage: Predicted damage (batch, num_elements)
        """
        # Handle time dimension if present
        if x.dim() == 3:
            x = x.mean(dim=1)  # Average over time steps
        
        # Encode
        encoded = self.encoder(x)
        
        # Multi-scale features
        feat_1 = self.feature_pyramid[0](encoded)
        feat_2 = self.feature_pyramid[1](feat_1)
        
        # Concatenate multi-scale features
        multi_scale = torch.cat([feat_1, feat_2], dim=-1)
        
        # Predict damage (before sigmoid for numerical stability)
        damage_logits = self.damage_head(multi_scale)
        damage = self.sigmoid(damage_logits)
        
        return damage
    
    def physics_loss(
        self,
        damage_pred: torch.Tensor,
        E: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute physics constraint losses.
        
        Args:
            damage_pred: Predicted damage (batch, num_elements)
            E: Young's modulus (batch,) - optional
            
        Returns:
            Dictionary of physics loss components
        """
        losses = {}
        
        if not self.use_physics:
            losses['total_physics_loss'] = torch.tensor(0.0, device=damage_pred.device)
            return losses
        
        # 1. Smoothness loss (adjacent elements should have similar damage)
        if hasattr(self, 'element_adjacency') and self.element_adjacency is not None:
            adj = self.element_adjacency
            # Weighted smoothness: penalize differences between connected elements
            diff = damage_pred.unsqueeze(-1) - damage_pred.unsqueeze(-2)
            smooth_loss = (adj * (diff ** 2)).mean()
            losses['smoothness_loss'] = smooth_loss
        
        # 2. Range constraint (damage should be in [0, 1])
        range_loss = torch.mean(
            torch.relu(damage_pred - 1.0) ** 2 + 
            torch.relu(-damage_pred) ** 2
        )
        losses['range_loss'] = range_loss
        
        # 3. Monotonicity with stiffness (higher damage → lower effective stiffness)
        if hasattr(self, 'stiffness_importance') and self.stiffness_importance is not None:
            # Expected relationship: damage ↑ → stiffness ↓
            # Penalize positive correlation between damage and stiffness
            damage_mean = damage_pred.mean(dim=0)  # (num_elements,)
            stiffness = self.stiffness_importance
            
            # Compute correlation
            if len(damage_mean) > 1:
                damage_centered = damage_mean - damage_mean.mean()
                stiffness_centered = stiffness - stiffness.mean()
                
                correlation = (damage_centered * stiffness_centered).sum()
                correlation = correlation / (damage_centered.std() * stiffness_centered.std() + 1e-8)
                
                # Penalize positive correlation
                monotonicity_loss = torch.relu(correlation)
                losses['monotonicity_loss'] = monotonicity_loss
        
        # 4. Sparsity (encourage few damaged elements - optional prior)
        sparsity_loss = damage_pred.mean()
        losses['sparsity_loss'] = sparsity_loss
        
        # Total physics loss
        total_physics = (
            0.1 * losses.get('smoothness_loss', 0) +
            1.0 * losses['range_loss'] +
            0.1 * losses.get('monotonicity_loss', 0) +
            0.01 * losses['sparsity_loss']
        )
        losses['total_physics_loss'] = total_physics
        
        return losses
    
    def compute_total_loss(
        self,
        damage_pred: torch.Tensor,
        damage_true: torch.Tensor,
        data_weight: float = 1.0,
        physics_weight: float = 0.1,
        E: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss with data and physics terms.
        
        Args:
            damage_pred: Predicted damage
            damage_true: True damage
            data_weight: Weight for data loss
            physics_weight: Weight for physics loss
            E: Young's modulus
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of loss components
        """
        # Data loss (MSE)
        data_loss = torch.nn.functional.mse_loss(damage_pred, damage_true)
        
        # Physics loss
        physics_losses = self.physics_loss(damage_pred, E)
        physics_loss = physics_losses['total_physics_loss']
        
        # Combined loss
        total_loss = data_weight * data_loss + physics_weight * physics_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'physics_loss': physics_loss.item(),
            **{k: v.item() if isinstance(v, torch.Tensor) else v 
               for k, v in physics_losses.items()}
        }
        
        return total_loss, loss_dict


class PINNEnsemble(nn.Module):
    """
    Ensemble of PINN models for improved robustness.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_models: int = 3,
        hidden_dims: List[int] = None
    ):
        super().__init__()
        
        self.models = nn.ModuleList([
            PINNDamagePredictorV2(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                dropout=0.3
            )
            for _ in range(num_models)
        ])
        
    def forward(self, x: torch.Tensor, E: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass returning ensemble mean."""
        predictions = [model(x, E) for model in self.models]
        return torch.stack(predictions).mean(dim=0)
    
    def predict_with_uncertainty(self, x: torch.Tensor, E: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with epistemic uncertainty estimate."""
        predictions = torch.stack([model(x, E) for model in self.models])
        mean = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        return mean, uncertainty
