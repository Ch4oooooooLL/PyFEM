# Coding Conventions

## Style Guidelines

### Python Version
- **Python 3.9+** required
- Modern syntax: `match/case`, `|` union types

### Import Order
```python
# 1. Standard library
import os
import sys
from typing import Dict, Tuple, List

# 2. Third-party
import numpy as np
import torch
import yaml

# 3. Local modules (relative within packages)
from core.node import Node
from core.material import Material
```

### Type Hints
Always use type hints for function signatures:

```python
# Good
def build_graph_info(element_conn: np.ndarray, num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    pass

# Modern union syntax (preferred)
def evaluate(...) -> Dict[str, float] | Tuple[Dict[str, float], np.ndarray]:
    pass

# Avoid (old syntax)
from typing import Union
def evaluate(...) -> Union[Dict, Tuple]:
    pass
```

---

## Naming Conventions

| Category | Convention | Example |
|----------|------------|---------|
| Classes | PascalCase | `Node`, `TrussElement2D`, `GTDamagePredictor` |
| Functions | snake_case | `get_local_stiffness`, `train_one_epoch` |
| Variables | snake_case | `element_conn`, `num_nodes` |
| Constants | SCREAMING_SNAKE_CASE | `ROOT_DIR`, `SCRIPT_DIR` |
| Private | _prefix | `_resolve_path`, `_set_global_determinism` |
| Abstract | PascalCase + implied type | `Element2D` |

---

## Class Design

### Abstract Base Classes
```python
from abc import ABC, abstractmethod

class Element2D(ABC):
    @abstractmethod
    def get_local_stiffness(self) -> np.ndarray:
        pass
```

### Properties for Computed Attributes
```python
@property
def length(self) -> float:
    return float(np.hypot(self.node2.x - self.node1.x, self.node2.y - self.node1.y))

@property
def angle(self) -> float:
    return float(np.arctan2(self.node2.y - self.node1.y, self.node2.x - self.node1.x))
```

---

## Error Handling

Use specific exceptions with meaningful messages:

```python
if not os.path.exists(gt_ckpt):
    raise FileNotFoundError(f"Checkpoint not found: {gt_ckpt}")

if output_dim != num_elements:
    raise ValueError(f"output_dim={output_dim} != num_elements={num_elements}")
```

---

## Code Patterns

### File Path Resolution
```python
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

def _resolve_path(base_dir: str, value: str) -> str:
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(base_dir, value))
```

### Device Handling
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

### Determinism
```python
def _set_global_determinism(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

### Configuration Loading
```python
import yaml

with open(config_path, "r", encoding="utf-8") as f:
    return yaml.safe_load(f) or {}
```

---

## Data Processing

### NumPy for Array Operations
```python
# Good
adj = np.eye(num_nodes)
edge_index = element_conn.copy()

# Matrix operations
K_global[np.ix_(dofs, dofs)] += k_global_element
```

### PyTorch for ML
```python
model = GTDamagePredictor(...).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))
```

---

## Documentation

### Comments
- **No comments unless explicitly required** (per AGENTS.md)
- Code should be self-explanatory through clear naming
- Docstrings for classes and public methods acceptable

### YAML Comments
Extensive comments in Chinese for configuration files:
```yaml
# 结构几何与材料配置文件
# 用于定义桁架/梁结构的节点坐标、单元连接、材料属性和边界条件

materials:  # 材料定义
  - name: "steel"      # 材料名称
    E: 2.0e+11         # 弹性模量 (Pa)
```

---

## Code Organization

### Within-Package Imports
```python
# Inside PyFEM_Dynamics/core/element.py
from core.node import Node      # Relative import
from core.material import Material
```

### Cross-Module Imports
```python
# From Deep_learning/ accessing PyFEM_Dynamics/
import sys
sys.path.append(os.path.join(ROOT_DIR, "PyFEM_Dynamics"))
from pipeline.data_gen import generate_dataset
```

---

## DOF Conventions

| Element Type | DOF per Node | Description |
|--------------|--------------|-------------|
| Truss | 2 | ux (horizontal), uy (vertical) |
| Beam | 3 | ux, uy, rz (rotation) |

Total DOFs calculated as: `num_nodes × dofs_per_node`
