# AGENTS.md - Agentic Coding Guidelines

This file provides guidance for agentic coding agents working in this repository.

## Project Overview

Python-based Finite Element Method (FEM) solver combined with deep learning modules for structural damage identification and condition prediction. Three main components:
- `PyFEM_Dynamics/` — FEM solver (core + solver + pipeline)
- `Deep_learning/` — GT (Graph Transformer) and PINN model training
- `Condition_prediction/` — Integrated pipeline for FEM + DL comparison

---

## Build/Lint/Test Commands

### Installation
```bash
pip install -r requirements.txt
pip install -r Deep_learning/requirements.txt
```

### Running the Project

**FEM static analysis:**
```bash
python PyFEM_Dynamics/main.py
```

**Generate training dataset:**
```bash
python PyFEM_Dynamics/pipeline/data_gen.py
```

**Train deep learning models:**
```bash
python Deep_learning/train.py --model gt --epochs 100
python Deep_learning/train.py --model pinn --epochs 100
python Deep_learning/train.py --model both --epochs 100
```

**Evaluate only (no training):**
```bash
python Deep_learning/train.py --model gt --eval_only
python Deep_learning/train.py --model pinn --eval_only
```

**Run condition prediction pipeline:**
```bash
python Condition_prediction/scripts/run_condition_prediction_cli.py --config condition_case.yaml
```

### Smoke Tests (No Formal Test Suite)

```bash
python PyFEM_Dynamics/main.py
python PyFEM_Dynamics/pipeline/data_gen.py
python Deep_learning/train.py --model gt --epochs 1
python Condition_prediction/scripts/run_condition_prediction_cli.py --config condition_case.yaml
```

### Single Test / Debugging

Since there's no pytest framework, use targeted execution:
```bash
# Test a single model for 1 epoch
python Deep_learning/train.py --model gt --epochs 1

# Test with specific config
python Deep_learning/train.py --config dataset_config.yaml --model gt --epochs 1

# Debug with smaller batch
python Deep_learning/train.py --model gt --epochs 1 --batch_size 8
```

---

## Code Style Guidelines

### General Principles
- Use Python 3.9+ features (type hints, match/case, etc.)
- No comments unless explicitly required by the user
- Follow existing code patterns in each module

### Imports

**Standard library first, then third-party, then local:**
```python
import os
import sys
import json
import argparse
from typing import Dict, Optional, Tuple
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import yaml

from data.dataset import FEMDataset
from models.gt_model import GTDamagePredictor
```

**Use explicit relative imports within packages:**
```python
from core.node import Node
from core.material import Material
from core.section import Section
```

### Type Hints

**Always use type hints for function signatures:**
```python
def build_graph_info(element_conn: np.ndarray, num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_type: str = 'gt',
) -> Dict[str, float]:
```

**Use modern Python union syntax:**
```python
# Good
def evaluate(..., return_raw: bool = False) -> Dict[str, float] | Tuple[Dict[str, float], np.ndarray, np.ndarray]:

# Avoid
from typing import Union
def evaluate(..., return_raw: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], np.ndarray, np.ndarray]]:
```

### Naming Conventions

- **Classes:** PascalCase (`Node`, `TrussElement2D`, `GTDamagePredictor`)
- **Functions/methods:** snake_case (`build_graph_info`, `train_one_epoch`)
- **Variables:** snake_case (`element_conn`, `num_nodes`, `val_loader`)
- **Constants:** SCREAMING_SNAKE_CASE (`ROOT_DIR`, `SCRIPT_DIR`)
- **Private methods:** prefix with underscore (`_resolve_path`, `_set_global_determinism`)

### Classes and OOP

**Use ABC for abstract base classes:**
```python
from abc import ABC, abstractmethod

class Element2D(ABC):
    @abstractmethod
    def get_local_stiffness(self) -> np.ndarray:
        pass
```

**Use properties for computed attributes:**
```python
@property
def length(self) -> float:
    return float(np.hypot(self.node2.x - self.node1.x, self.node2.y - self.node1.y))
```

### Error Handling

**Use specific exceptions and meaningful messages:**
```python
if not os.path.exists(gt_ckpt):
    raise FileNotFoundError(f"Checkpoint not found for eval_only mode: {gt_ckpt}")

if output_dim != num_elements:
    raise ValueError(
        f"Dataset output_dim={output_dim} does not match structure num_elements={num_elements}"
    )
```

### Data Processing

**Use NumPy for array operations:**
```python
adj = np.eye(num_nodes)
edge_index = element_conn.copy()
for n1, n2 in element_conn:
    adj[n1, n2] = 1
    adj[n2, n1] = 1
```

**Use PyTorch for ML models:**
```python
model = GTDamagePredictor(...).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))
```

### Configuration

**Use YAML for configuration files:**
- `structure.yaml` — Node/element topology, boundary conditions
- `dataset_config.yaml` — Load generation params, damage ranges, time settings
- `condition_case.yaml` — Load case, damage case, inference settings

**Load config with yaml.safe_load:**
```python
with open(config_path, "r", encoding="utf-8") as f:
    return yaml.safe_load(f) or {}
```

### GPU/CUDA Handling

**Check availability before using CUDA:**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

### Determinism

**Set global determinism for reproducibility:**
```python
def _set_global_determinism(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

### File Paths

**Use os.path for path operations:**
```python
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

def _resolve_path(base_dir: str, value: str) -> str:
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(base_dir, value))
```

### Docstrings

**Use Google-style docstrings when documentation is needed:**
```python
def build_graph_info(element_conn: np.ndarray, num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建节点邻接矩阵和边索引
    
    Returns:
        adj: (num_nodes, num_nodes)
        edge_index: (num_elements, 2)
    """
```

---

## Key Data Formats

- `structure.yaml` — Node/element topology, boundary conditions
- `dataset_config.yaml` — Load generation params, damage ranges, time settings
- `condition_case.yaml` — Load case, damage case, inference settings
- `dataset/train.npz` — Keys: `load`, `E`, `disp`, `stress`, `damage`

---

## Important Notes

- Generated artifacts (dataset/, postprocess_results/, checkpoints/, outputs/) are gitignored
- DOF assumptions: 2 dof/node for truss (ux, uy), 3 dof/node for beam (ux, uy, rz)
- Default checkpoints: `Deep_learning/checkpoints/gt_best.pth`, `pinn_best.pth`
- Training output dimension must match structure num_elements
