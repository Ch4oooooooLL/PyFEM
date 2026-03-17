# Integrations

## Overview
This codebase is primarily self-contained with minimal external service dependencies. It focuses on scientific computing and machine learning workflows rather than web service integrations.

---

## External Libraries

### Core Scientific Stack
| Library | Purpose | Key Usage Locations |
|---------|---------|---------------------|
| NumPy | Array operations, matrix math | `PyFEM_Dynamics/core/`, `solver/` |
| SciPy | Sparse matrices, LU solvers | `solver/assembler.py`, `solver/integrator.py` |
| PyTorch | Neural networks, GPU compute | `Deep_learning/models/`, `train.py` |

### Visualization
| Library | Purpose | Key Usage Locations |
|---------|---------|---------------------|
| Matplotlib | Static plots, deformation viz | `postprocess/plotter.py`, `notebooks/` |
| Seaborn | Statistical plots, heatmaps | `Deep_learning/utils/visualization.py` |

### Configuration
| Library | Purpose | Key Usage Locations |
|---------|---------|---------------------|
| PyYAML | Config file parsing | `core/io_parser.py`, all pipeline modules |

---

## Data Flow

### Training Data Pipeline
```
structure.yaml + dataset_config.yaml
    ↓
data_gen.py (FEM simulation)
    ↓
dataset/train.npz
    ↓
FEMDataset (PyTorch Dataset)
    ↓
DataLoader → Model Training
```

### Inference Pipeline
```
condition_case.yaml
    ↓
run_condition_prediction_cli.py
    ↓
ConditionPipeline
    ↓
FEM Solver + DL Model Inference
    ↓
Comparison Results
```

---

## File System Dependencies

### Input Files (User-Provided)
- `structure.yaml` — Structure definition (required)
- `dataset_config.yaml` — Generation parameters (optional, defaults available)
- `condition_case.yaml` — Case configuration (for inference)

### Generated Artifacts (Gitignored)
- `dataset/*.npz` — Training datasets
- `postprocess_results/` — Analysis outputs
- `Deep_learning/checkpoints/*.pth` — Model weights
- `outputs/` — General output directory

### Default Checkpoints
- `Deep_learning/checkpoints/gt_best.pth` — Graph Transformer best model
- `Deep_learning/checkpoints/pinn_best.pth` — PINN best model

---

## No External APIs

This codebase does not integrate with:
- REST APIs
- Databases (SQL/NoSQL)
- Authentication providers
- Cloud services
- Message queues
- Webhooks

All dependencies are installable via pip from PyPI.

---

## Reproducibility Features

### Deterministic Mode
```python
def _set_global_determinism(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

### Version Pinning
- Requirements files specify minimum versions
- No strict upper bounds (may introduce breaking changes)
