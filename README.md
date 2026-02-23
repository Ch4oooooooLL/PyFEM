# PyFEM-Dynamics 二维结构动力学有限元程序

## 1. 简介
本程序是一个使用 Python 编写的二维有限元求解器，主要用于学习计算桁架和梁结构的静力与动力响应。本程序为数据驱动结构健康监测研究提供基于物理模型的仿真数据。当前仓库由两部分组成：

1. `PyFEM_Dynamics/`：有限元建模、静/动力学求解、后处理可视化。
2. `deep_learning/`：基于仿真数据的损伤识别模型训练（LSTM / PINN）与结果分析。

## 2. 力学原理与数值方法

### 2.1 单元列式
程序目前实现了基于欧拉-伯努利（Euler-Bernoulli）假设的平面梁单元以及二维平面拉压桁架单元（Truss2D），适用于线弹性、小变形条件下的结构受力分析。以二维桁架单元为例：

根据虚功原理，在局部坐标系下，单根杆件的刚度矩阵 $\mathbf{k}^e$ 和一致质量矩阵 $\mathbf{m}^e$ 推导结果如下：

$$
\mathbf{k}^e = \frac{EA}{L}\begin{bmatrix} 1 & -1 \\\\ -1 & 1 \end{bmatrix}, \quad \quad \mathbf{m}^e = \frac{\rho A L}{6}\begin{bmatrix} 2 & 1 \\\\ 1 & 2 \end{bmatrix}
$$

在结构动力学计算中，为了提高计算效率，程序在 `solver/assembler.py` 中提供了集中质量矩阵（Lumped Mass Matrix）的选项对质量矩阵进行对角化：

$$
\mathbf{m}_{lumped}^e = \frac{\rho A L}{2}\begin{bmatrix} 1 & 0 \\\\ 0 & 1 \end{bmatrix}
$$

单元矩阵建立后，通过坐标转换矩阵 $\mathbf{T}$（其中包含方向余弦 $c = \cos\theta, s = \sin\theta$），将局部刚度矩阵转换到全局笛卡尔坐标系：

$$
\mathbf{K}^e = \mathbf{T}^\top \mathbf{k}^e \mathbf{T}
$$

### 2.2 全局组装与边界条件
全局刚度矩阵 $\mathbf{K}$ 和质量矩阵 $\mathbf{M}$ 根据直接刚度法（Direct Stiffness Method）进行组装，将各个单元的矩阵依据拓扑关系叠加到系统总体矩阵中：

$$
\mathbf{K} = \sum_{e=1}^{nel} \mathbf{L}_e^\top \mathbf{K}^e \mathbf{L}_e
$$

在 `solver/boundary.py` 中，程序提供了两种处理本质边界条件（位移边界条件，如固定支座）的方法：
1. **划零划一法（Zero-One Substitution Method）**：常用于静力学求解，通过修改方程系统的对应行和列，保持矩阵对称性。
2. **大数法（Penalty Method / 乘大数法）**：在受约束节点对应的刚度矩阵主对角线元素上乘以一个极大的惩罚因子（如 $\alpha \approx 10^{15}$），并在等效节点载荷向量的相应位置乘以同样的因子 $\alpha \times \bar{u}$。这种方法能在不改变全局刚度方程维度的前提下数值近似求解边界条件。

### 2.3 结构动力学时间积分
结构多自由度系统的动态运动方程为：

$$
\mathbf{M}\ddot{\mathbf{u}}(t) + \mathbf{C}\dot{\mathbf{u}}(t) + \mathbf{K}\mathbf{u}(t) = \mathbf{F}(t)
$$

程序采用 **Rayleigh 阻尼（比例阻尼）** 模型构建全局阻尼矩阵： $\mathbf{C} = \alpha \mathbf{M} + \beta \mathbf{K}$。

时间积分求解器采用了 **Newmark-$\beta$ 法** 中的平均加速度法格式（其中 $\gamma = \frac{1}{2}, \beta = \frac{1}{4}$），该格式对于线性系统是无条件稳定的，截断误差为 $O(\Delta t^2)$。在 `solver/integrator.py` 的实现中，每个时间步的推进转化为求解等效代数方程组问题，其等效刚度矩阵 $\mathbf{\hat{K}}$ 为：

$$
\mathbf{\hat{K}} = \mathbf{K} + a_0 \mathbf{M} + a_1 \mathbf{C}  \quad \Biggl( \text{其中 } a_0 = \frac{1}{\beta \Delta t^2}, \ a_1 = \frac{\gamma}{\beta \Delta t} \Biggr)
$$

### 2.4 参数化分析与数据批量生成
`pipeline/data_gen.py` 模块提供批量动力分析能力。程序通过对选定单元弹性模量 $E$ 进行折减（例如乘以 $0.5 \sim 0.9$）模拟损伤，然后对每个样本执行时程积分，输出全节点全时刻位移、单元应力历史和损伤标签。对桁架模型，von Mises 应力采用一维近似 $\sigma_{vm}=|\sigma_{axial}|$。

## 3. 使用方法与输入格式

本项目采用基于 YAML 的统一配置体系，确立了以 `structure.yaml` 为核心的输入规范，彻底弃用了旧版的 CSV/TXT 多文件模式。

### 3.1 结构定义文 (`structure.yaml`)

用于完整定义有限元模型的几何拓扑、材料库及边界条件。

```yaml
metadata:
  description: "2D Truss Structure"
  num_nodes: 5
  num_elements: 7
  num_dofs: 10
  dofs_per_node: 2

# 材料库定义
materials:
  - name: "steel"
    E: 2.0e+11    # 弹性模量 (Pa)
    rho: 7850.0   # 密度 (kg/m^3)
    nu: 0.3       # 泊松比

nodes:
  - id: 0
    coords: [0.0, 0.0]
  # ...

elements:
  - id: 0
    nodes: [0, 1]
    material: "steel"  # 引用材料名
    A: 0.005           # 截面积 (m^2)
    I: 0.0             # 惯性矩 (m^4)
  # ...

boundary:
  - node_id: 0
    constraints: [ux, uy]  # 固定端
```

### 3.2 数据集配置 (`dataset_config.yaml`)

用于批量生成动力响应数据集。

```yaml
structure_file: structure.yaml
output_file: dataset/train.npz

time:
  dt: 0.01          # 时间步长
  total_time: 2.0   # 总时长

generation:
  num_samples: 1000 # 生成样本总数
  random_seed: 42   # 随机种子

damage:
  enabled: true
  min_damaged_elements: 1
  max_damaged_elements: 3
  reduction_range: [0.5, 0.9] # 损伤衰减系数范围

load_generation:
  mode: "random"
  loads:
    - node_id: 3
      dof: "fx"
      pattern: "pulse"
      F0_range: [8000, 12000]
      t_start_range: [0.0, 0.05]
      t_end_range: [0.15, 0.25]
```

### 3.3 支持的载荷模式

在 `dataset_config.yaml` 的 `load_generation` 部分，支持以下模式：

| 模式 | 公式 | 参数说明 |
|------|------|------|
| `pulse` | F(t) = F0 (t_start ≤ t < t_end) | `F0_range`, `t_start_range`, `t_end_range` |
| `harmonic` | F(t) = F0 * sin(2πft) | `F0_range`, `freq`, `duration` |
| `half_sine` | F(t) = F0 * sin(π(t-t_s)/(t_e-t_s)) | `F0_range`, `t_start_range`, `t_end_range` |
| `ramp` | F(t) = F0 * (t-t_s)/t_ramp | `F0`, `t_start`, `t_ramp` |
| `gaussian` | 脉冲型高斯载荷 | `F0_range`, `t0_range`, `sigma_range` |

### 3.4 运行指南

程序的主要功能通过以下入口访问：

```bash
# 1. 静力验证（检查模型正确性）
python PyFEM_Dynamics/main.py

# 2. 批量生成动力响应数据集（用于深度学习）
python PyFEM_Dynamics/pipeline/data_gen.py

# 3. 训练损伤识别模型（LSTM / PINN）
python deep_learning/train.py --model both --epochs 100

# 4. 生成可视化评估报告
python deep_learning/utils/visualization.py --checkpoints_dir deep_learning/checkpoints --aggregate_all
```

### 3.5 输出格式说明

- **数据集 (`dataset/train.npz`)**:
  - `load`: (N, T, DOF) - 载荷时程
  - `E`: (N, elem) - 弹性模量
  - `disp`: (N, T, DOF) - 位移响应
  - `stress`: (N, T, elem) - 单元应力响应
  - `damage`: (N, elem) - 损伤系数 (1.0表示无损)
- **元数据 (`metadata.json`)**: 记录数据集对应的物理参数及维度信息。
- **训练成果**: 包含 `.pth` 权重文件、`.json` 指标文件及自动生成的 `figures/` 可视化报告。

## 4. 算例与结果展示

程序内置了绘图接口，支持对输出的节点位移、单元力以及时间历史进行可视化评估：

**1. 静力分析演示：桁架受载变形与轴力分布**
模型计算后，利用内置方法通过不同颜色深浅或色系隐式映射了每个单元内部的轴引力或压力，并可以在图形上将小变形按照给定缩放因子进行放大，直观呈现节点变位：
![static_deformation](docs/images/supervisor/static_deformation.png)

**2. 动力参数化抽样：位移-应力数据集与 VM 云图**
批量流程会输出每样本位移/应力数据，并可导出 VM 全时程抽帧云图用于可视化检查：

![vm_cloud_frame](docs/images/supervisor/vm_cloud_frame_100.png)

**3. 深度学习训练效果展示（LSTM / PINN）**

LSTM 训练曲线：

![lstm_history](docs/images/supervisor/lstm_history.png)

PINN 训练曲线：

![pinn_history](docs/images/supervisor/pinn_history.png)

测试集综合指标对比：

![model_comparison](docs/images/supervisor/model_comparison.png)

跨运行稳定性（95% CI）：

![run_stability](docs/images/supervisor/run_stability.png)

