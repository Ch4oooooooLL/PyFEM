# Technology Stack

## Overview
Python-based Finite Element Method (FEM) solver combined with deep learning for structural damage identification.

---

## Core Technologies

### Programming Language
- **Python 3.9+** — Primary development language
- Type hints extensively used throughout codebase
- Modern Python features: match/case, `|` union syntax

### Core Dependencies

#### Scientific Computing
- **NumPy** (>=1.21.0) — Array operations, linear algebra
- **SciPy** (>=1.10.0) — Sparse matrices, solvers (`scipy.sparse`, `scipy.linalg`)

#### Deep Learning
- **PyTorch** (>=2.0.0) — Neural network framework
  - GPU/CUDA support with auto-detection
  - Custom nn.Module implementations for GT and PINN models

#### Visualization
- **Matplotlib** (>=3.5.0) — Plotting and charts
- **Seaborn** (>=0.11.0) — Statistical visualization

#### Configuration
- **PyYAML** (>=6.0) — YAML configuration file parsing

#### Utilities
- **tqdm** (>=4.65.0) — Progress bars for training loops
- **ipywidgets** (>=8.0.0) — Jupyter interactive widgets

---

## Architecture Patterns

### FEM Solver (`PyFEM_Dynamics/`)
- Object-oriented design with abstract base classes
- Sparse matrix operations for large systems
- Newmark-β implicit time integration

### Deep Learning (`Deep_learning/`)
- PyTorch nn.Module subclasses
- Graph Transformer with multi-head self-attention
- Physics-Informed Neural Network (PINN) architecture

### Pipeline Architecture
- Configuration-driven workflows via YAML
- Modular pipeline components with clear data flow

---

## Configuration Files

| File | Purpose |
|------|---------|
| `structure.yaml` | Node/element topology, boundary conditions |
| `dataset_config.yaml` | Load generation, damage ranges, time settings |
| `condition_case.yaml` | Load case, damage case, inference settings |
| `load_template.yaml` | Load pattern templates |

---

## Key File Formats

- **NPZ** — NumPy compressed format for training datasets (`dataset/train.npz`)
  - Keys: `load`, `E`, `disp`, `stress`, `damage`
- **YAML** — Human-readable configuration
- **PTH** — PyTorch model checkpoints

---

## GPU/CUDA Support

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- Automatic device detection
- Deterministic training mode available
- CUDA manual seeding for reproducibility
