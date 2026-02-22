# PyFEM-Dynamics 输入文件格式说明

项目支持两套输入体系：

## 体系一：传统三文件模式（兼容旧版）

### 1. 材料文件 (`materials.csv`)
格式:
```csv
id,E,rho
0,200000000000.0,7850.0
```

### 2. 结构文件 (`structure_input.txt`)
仅用于结构与边界条件定义，支持如下指令:
- `NODE, node_id, x, y`
- `ELEM, elem_id, type, node1_id, node2_id, mat_id, A, I`
- `BC, node_id, local_dof, value`
- `CONFIG, key, value` (可选)

### 3. 静载文件 (`static_loads.txt`)
- `SLOAD, node_id, Fx, Fy`

### 4. 动载文件 (`dynamic_loads.txt`)
- `CONFIG, key, value`
- `DLOAD_RANGE, node_id, fx_min, fx_max, fy_min, fy_max, t_start_min, t_start_max, t_end_min, t_end_max`

---

## 体系二：YAML统一配置模式（推荐用于深度学习）

### 1. 结构定义文件 (`structure.yaml`)

```yaml
metadata:
  description: "2D Truss Structure"
  num_nodes: 5
  num_elements: 7
  num_dofs: 10
  dofs_per_node: 2

nodes:
  - id: 0
    coords: [0.0, 0.0]
  - id: 1
    coords: [2.0, 0.0]
  # ...

elements:
  - id: 0
    nodes: [0, 1]
    E: 2.0e+11
    rho: 7850.0
    A: 0.005
    I: 0.0
  # ...

boundary:
  - node_id: 0
    constraints: [ux, uy]  # 固定端
  - node_id: 2
    constraints: [uy]      # 竖向约束
```

### 2. 数据集配置文件 (`dataset_config.yaml`)

```yaml
structure_file: structure.yaml
output_file: dataset/train.npz

time:
  dt: 0.01
  total_time: 2.0

generation:
  num_samples: 1000
  random_seed: 42

damage:
  enabled: true
  min_damaged_elements: 1
  max_damaged_elements: 3
  reduction_range: [0.5, 0.9]

load_generation:
  mode: "random"
  loads:
    - node_id: 3
      dof: "fx"
      pattern: "pulse"
      F0_range: [8000, 12000]
      t_start_range: [0.0, 0.05]
      t_end_range: [0.15, 0.25]
    - node_id: 4
      dof: "fy"
      pattern: "half_sine"
      F0_range: [-15000, -8000]
      t_start_range: [0.1, 0.3]
      t_end_range: [0.35, 0.6]

damping:
  alpha: 0.1
  beta: 0.01
```

### 3. 支持的载荷模式 (`load_template.yaml`)

| 模式 | 公式 | 参数 |
|------|------|------|
| `pulse` | F(t) = F0, t_start ≤ t < t_end | F0_range, t_start_range, t_end_range |
| `harmonic` | F(t) = F0 * sin(2πft) | F0_range, freq, duration |
| `half_sine` | F(t) = F0 * sin(π(t-t_start)/(t_end-t_start)) | F0_range, t_start_range, t_end_range |
| `ramp` | F(t) = F0 * (t-t_start)/t_ramp | F0, t_start, t_ramp |
| `gaussian` | F(t) = F0 * exp(-0.5((t-t0)/σ)²) | F0_range, t0_range, sigma_range |

---

## 输出格式

### NPZ数据集格式

运行 `PyFEM_Dynamics/pipeline/data_gen.py` 后，输出 `train.npz`:

```python
data = np.load('dataset/train.npz')

# 数据维度说明
data['load']    # (N, T, num_dofs)    - 载荷时程矩阵
data['E']       # (N, num_elements)   - 弹性模量（含损伤）
data['disp']    # (N, T, num_dofs)    - 位移响应
data['stress']  # (N, T, num_elements) - 应力响应(von Mises)
data['damage']  # (N, num_elements)   - 损伤程度 (1.0=无损, <1=损伤)
```

### 元数据文件 (`metadata.json`)

```json
{
  "structure_file": "structure.yaml",
  "num_samples": 1000,
  "num_nodes": 5,
  "num_elements": 7,
  "num_dofs": 10,
  "dt": 0.01,
  "total_time": 2.0,
  "num_steps": 201
}
```

---

## 深度学习训练示例

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class FEMDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.loads = torch.from_numpy(data['load']).float()
        self.disp = torch.from_numpy(data['disp']).float()
        self.damage = torch.from_numpy(data['damage']).float()
    
    def __len__(self):
        return len(self.loads)
    
    def __getitem__(self, idx):
        # 任务: 载荷+响应 -> 损伤预测
        x = torch.cat([
            self.loads[idx].flatten(),
            self.disp[idx].flatten()
        ])
        y = self.damage[idx]
        return x, y

# 使用
dataset = FEMDataset('dataset/train.npz')
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## 入口文件

- **静力验证**: `PyFEM_Dynamics/main.py`（使用传统格式）
- **数据集生成**: `PyFEM_Dynamics/pipeline/data_gen.py`（使用YAML格式）
