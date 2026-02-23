# PyFEM-Dynamics 二维结构动力学有限元程序

## 1. 简介
本程序是一个使用 Python 编写的二维有限元求解器，主要用于学习计算桁架和梁结构的静力与动力响应。本程序为数据驱动结构健康监测研究提供基于物理模型的仿真数据。当前仓库由两部分组成：

1. `PyFEM_Dynamics/`：有限元建模、静/动力学求解、后处理可视化。
2. `deep_learning/`：基于仿真数据的损伤识别模型训练（**Graph Transformer** / PINN）与结果分析。

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

### 2.2 全局组装与数值稳定性
全局刚度矩阵 $\mathbf{K}$ 和质量矩阵 $\mathbf{M}$ 根据直接刚度法（Direct Stiffness Method）进行组装。对于动力学边界处理，程序采用 **划零划一法（Zero-One Substitution Method）** 处理本质边界条件。相比于罚函数法，该方法在时程积分中能完全解耦边界自由度，避免了数值刚度过大导致的计算发散：

$$
\mathbf{K}_{ij} = \delta_{ij}, \quad \mathbf{F}_i = \bar{u}_i \quad (\text{if DOF } i \text{ is constrained})
$$

### 2.3 结构动力学时间积分
结构多自由度系统的动态运动方程为：

$$
\mathbf{M}\ddot{\mathbf{u}}(t) + \mathbf{C}\dot{\mathbf{u}}(t) + \mathbf{K}\mathbf{u}(t) = \mathbf{F}(t)
$$

程序采用 **Rayleigh 阻尼（比例阻尼）** 模型构建全局阻尼矩阵： $\mathbf{C} = \alpha \mathbf{M} + \beta \mathbf{K}$。时间积分求解器采用了 **Newmark-$\beta$ 法**（$\gamma = \frac{1}{2}, \beta = \frac{1}{4}$），确保线性系统的无条件稳定。

### 2.4 参数化分析与大规模数据生成
`pipeline/data_gen.py` 模块提供批量动力分析能力。程序通过对选定单元弹性模量 $E$ 进行折减模拟损伤。为了支持深度学习（特别是 Graph Transformer），本程序支持生成 **10,000** 样本量级的物理增强数据集。

## 3. 深度学习架构：Graph Transformer (GT)

项目已从传统的 LSTM 序列模型迁移至更适应物理拓扑的 **Graph Transformer** 架构。

### 3.1 图拓扑建模
我们将有限元节点建模为图的顶点（Nodes），将单元（杆件）建模为图的边（Edges）：
- **节点特征 (Node Features)**: 坐标 $(x, y)$ 及位移响应时程 $(u_x, u_y)_t$。
- **空间注意力 (Spatial Attention)**: 利用图注意力机制（GAT）计算节点间的信息传递权重，捕捉力学信号在物理结构中的全局传播规律。
- **损伤预测 (Edge Prediction)**: 基于单元两端节点的特征聚合，预测该单元的损伤系数（1.0为无损，0.5-0.9表示不同程度损伤）。

## 4. 使用方法与运行指南

### 4.1 配置文件说明
- `structure.yaml`: 定义有限元模型的几何拓扑、材料库及边界条件。
- `dataset_config.yaml`: 控制大规模数据生成的规模、损伤模式及随机载荷参数。

### 4.2 运行步骤

```bash
# 1. 静力验证
python PyFEM_Dynamics/main.py

# 2. 生成 10,000 规模动力响应数据集
python PyFEM_Dynamics/pipeline/data_gen.py --config dataset_config.yaml

# 3. 训练 Graph Transformer 模型 (GT)
python deep_learning/train.py --model gt --epochs 100 --batch_size 128

# 4. 训练 PINN (物理信息神经网络) 模型
python deep_learning/train.py --model pinn --epochs 100
```

## 5. 结果展示

程序内置了高精度的多物理量可视化接口：

**1. 动力可视化：von Mises 应力云图 (修正缩放因子)**
动态展示结构在时程载荷下的应力分布。通过优化的坐标变换逻辑，准确叠加变形后的几何形态：
![vm_cloud_frame](docs/images/supervisor/vm_cloud_frame_100.png)

**2. 深度学习迁移：GT 训练监控**
Graph Transformer 在复杂拓扑识别任务中具有更佳的收敛速度和物理一致性。
![gt_metrics](docs/images/supervisor/model_comparison.png) 
*(注：GT 指标优于传统的序列建模方法)*

---
**致谢**: 本项目为工程力学本科生力学数值实验与 AI+Science 交叉研究成果。

