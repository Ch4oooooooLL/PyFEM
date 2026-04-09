"""
PyFEM-Dynamics Deep Learning Module
用于结构损伤识别的深度学习模型
"""

from .utils.metrics import compute_metrics
from .utils.visualization import TrainingVisualizer, visualize_results

__all__ = ['compute_metrics', 'TrainingVisualizer', 'visualize_results']

# Optional imports: allow postprocessing utilities to run even when torch is absent.
try:
    from .models.pinn_model import PINNDamagePredictor
    from .models.pinn_model_v2 import PINNDamagePredictorV2
    from .data.dataset import FEMDataset

    __all__.extend([
        'PINNDamagePredictor',
        'PINNDamagePredictorV2',
        'FEMDataset',
    ])
except Exception:
    pass
