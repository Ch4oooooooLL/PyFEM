"""
Adaptive loss weighting module for PINN training.

Implements GradNorm-style adaptive weighting to balance data and physics loss terms.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class AdaptivePINNLoss(nn.Module):
    """
    Adaptive loss weighting for Physics-Informed Neural Networks.
    
    Uses GradNorm-inspired approach to dynamically balance data loss and physics loss
    based on their relative gradient magnitudes and training rates.
    
    Args:
        initial_data_weight: Initial weight for data loss term
        initial_physics_weight: Initial weight for physics loss term
        adaptive: Whether to use adaptive weighting (True) or static (False)
        temperature: Temperature parameter for weight updates (lower = slower adaptation)
        alpha: GradNorm alpha parameter controlling relative training rates
        min_weight: Minimum weight value to prevent collapse
        max_weight: Maximum weight value to prevent explosion
    """
    
    def __init__(
        self,
        initial_data_weight: float = 1.0,
        initial_physics_weight: float = 0.1,
        adaptive: bool = True,
        temperature: float = 0.1,
        alpha: float = 1.5,
        min_weight: float = 0.01,
        max_weight: float = 100.0
    ):
        super().__init__()
        
        self.adaptive = adaptive
        self.temperature = temperature
        self.alpha = alpha
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Learnable log-weights for numerical stability
        if adaptive:
            self.log_data_weight = nn.Parameter(
                torch.tensor([torch.log(torch.tensor(initial_data_weight))], dtype=torch.float32)
            )
            self.log_physics_weight = nn.Parameter(
                torch.tensor([torch.log(torch.tensor(initial_physics_weight))], dtype=torch.float32)
            )
        else:
            self.register_buffer('data_weight', torch.tensor(initial_data_weight))
            self.register_buffer('physics_weight', torch.tensor(initial_physics_weight))
        
        # Loss history for tracking
        self.register_buffer('loss_history', torch.zeros(100, 2))  # [data, physics]
        self.history_ptr = 0
        
        # Statistics tracking
        self.register_buffer('grad_norm_data', torch.tensor(0.0))
        self.register_buffer('grad_norm_physics', torch.tensor(0.0))
        
    @property
    def data_weight(self) -> torch.Tensor:
        """Get current data loss weight."""
        if self.adaptive:
            return torch.exp(self.log_data_weight).clamp(self.min_weight, self.max_weight)
        return self._buffers['data_weight']
    
    @property
    def physics_weight(self) -> torch.Tensor:
        """Get current physics loss weight."""
        if self.adaptive:
            return torch.exp(self.log_physics_weight).clamp(self.min_weight, self.max_weight)
        return self._buffers['physics_weight']
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        physics_residual: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted total loss.
        
        Args:
            predictions: Model predictions (batch, num_elements)
            targets: Ground truth targets (batch, num_elements)
            physics_residual: Physics constraint residual (batch, num_elements) or dict of residuals
            model: The model being trained (needed for adaptive weighting with GradNorm)
            
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary containing individual loss terms and current weights
        """
        # Data loss (MSE)
        data_loss = torch.nn.functional.mse_loss(predictions, targets)
        
        # Physics loss
        if physics_residual is None:
            physics_loss = torch.tensor(0.0, device=predictions.device)
        elif isinstance(physics_residual, dict):
            physics_loss = sum(v for v in physics_residual.values() if isinstance(v, torch.Tensor))
        else:
            physics_loss = physics_residual.mean()
        
        # Get current weights
        w_data = self.data_weight
        w_physics = self.physics_weight
        
        # Compute weighted loss
        if self.adaptive and model is not None and self.training:
            # GradNorm-style adaptive weighting
            total_loss = self._gradnorm_loss(
                data_loss, physics_loss, w_data, w_physics, model
            )
        else:
            # Simple weighted sum
            total_loss = w_data * data_loss + w_physics * physics_loss
        
        # Update history
        with torch.no_grad():
            idx = self.history_ptr % 100
            self.loss_history[idx] = torch.tensor([data_loss.item(), physics_loss.item()])
            self.history_ptr += 1
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'physics_loss': physics_loss.item(),
            'data_weight': w_data.item(),
            'physics_weight': w_physics.item()
        }
        
        return total_loss, loss_dict
    
    def _gradnorm_loss(
        self,
        data_loss: torch.Tensor,
        physics_loss: torch.Tensor,
        w_data: torch.Tensor,
        w_physics: torch.Tensor,
        model: nn.Module
    ) -> torch.Tensor:
        """
        GradNorm: Gradient Normalization for Adaptive Loss Balancing.
        
        Reference: Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing"
        """
        # Get shared parameters (last layer weights as proxy)
        shared_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and 'encoder' in name:
                shared_params.append(param)
        
        if not shared_params:
            # Fallback to simple weighted sum if no shared params found
            return w_data * data_loss + w_physics * physics_loss
        
        # Compute gradients for each loss term w.r.t. shared parameters
        grads_data = torch.autograd.grad(
            data_loss, shared_params, retain_graph=True, create_graph=True
        )
        grads_physics = torch.autograd.grad(
            physics_loss, shared_params, retain_graph=True, create_graph=True
        )
        
        # Compute gradient norms
        grad_norm_data = torch.stack([g.norm() for g in grads_data]).mean()
        grad_norm_physics = torch.stack([g.norm() for g in grads_physics]).mean()
        
        # Store for monitoring
        self.grad_norm_data = grad_norm_data.detach()
        self.grad_norm_physics = grad_norm_physics.detach()
        
        # GradNorm: balance gradient magnitudes
        # Target: grad_norm_i / w_i = constant for all tasks
        avg_grad_norm = (grad_norm_data + grad_norm_physics) / 2
        
        # Relative inverse training rates
        # Using loss history to compute rate of change
        if self.history_ptr >= 2:
            recent = self.loss_history[max(0, (self.history_ptr - 10) % 100):self.history_ptr % 100]
            if len(recent) >= 2:
                # Compute relative rate (simplified)
                loss_ratio = recent[-1] / (recent[0] + 1e-8)
                r_data = loss_ratio[0]
                r_physics = loss_ratio[1]
            else:
                r_data = r_physics = 1.0
        else:
            r_data = r_physics = 1.0
        
        # Target gradient norms
        target_norm_data = avg_grad_norm * (r_data ** self.alpha)
        target_norm_physics = avg_grad_norm * (r_physics ** self.alpha)
        
        # GradNorm loss encourages gradient norms to match targets
        grad_norm_loss = torch.abs(grad_norm_data - target_norm_data) + \
                        torch.abs(grad_norm_physics - target_norm_physics)
        
        # Total loss with GradNorm regularization
        total_loss = w_data * data_loss + w_physics * physics_loss + self.temperature * grad_norm_loss
        
        return total_loss
    
    def get_weight_history(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get history of loss terms for analysis."""
        if self.history_ptr == 0:
            return torch.tensor([]), torch.tensor([])
        
        valid_history = self.loss_history[:min(self.history_ptr, 100)]
        return valid_history[:, 0], valid_history[:, 1]
    
    def reset_weights(self, data_weight: float = 1.0, physics_weight: float = 0.1):
        """Reset weights to initial values."""
        if self.adaptive:
            with torch.no_grad():
                self.log_data_weight.fill_(torch.log(torch.tensor(data_weight)))
                self.log_physics_weight.fill_(torch.log(torch.tensor(physics_weight)))
        else:
            self.data_weight.fill_(data_weight)
            self.physics_weight.fill_(physics_weight)
        self.history_ptr = 0


class UncertaintyAdaptiveLoss(nn.Module):
    """
    Uncertainty-weighted adaptive loss (Kendall et al. 2018).
    
    Learns task-dependent uncertainty to weight losses.
    """
    
    def __init__(
        self,
        initial_data_weight: float = 1.0,
        initial_physics_weight: float = 0.1
    ):
        super().__init__()
        
        # Log variance for each task (learnable)
        self.log_sigma_data = nn.Parameter(torch.zeros(1))
        self.log_sigma_physics = nn.Parameter(torch.zeros(1))
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        physics_residual: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute uncertainty-weighted loss.
        
        Loss = 1/(2*sigma^2) * L_data + 1/(2*sigma^2) * L_physics + log(sigma)
        """
        # Data loss
        data_loss = torch.nn.functional.mse_loss(predictions, targets)
        
        # Physics loss
        if physics_residual is None:
            physics_loss = torch.tensor(0.0, device=predictions.device)
        elif isinstance(physics_residual, dict):
            physics_loss = sum(v for v in physics_residual.values() if isinstance(v, torch.Tensor))
        else:
            physics_loss = physics_residual.mean()
        
        # Uncertainty-weighted losses
        precision_data = torch.exp(-self.log_sigma_data)
        precision_physics = torch.exp(-self.log_sigma_physics)
        
        total_loss = (
            precision_data * data_loss +
            precision_physics * physics_loss +
            self.log_sigma_data +
            self.log_sigma_physics
        )
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'physics_loss': physics_loss.item(),
            'sigma_data': torch.exp(self.log_sigma_data).item(),
            'sigma_physics': torch.exp(self.log_sigma_physics).item()
        }
        
        return total_loss, loss_dict
