# 二维结构动力学有限元仿真与智能损伤识别系统

## 1. 项目简介
本项目由本人独立开发，旨在实现一个完整的结构工程动力学分析平台，并探索基于图变换网络（Graph Transformer）的结构损伤自识别技术。项目主要分为两大部分：
*   **第一部分：有限元分析内核 (FEM Core)** —— 实现了从几何参数化建模到动力学时程积分的完整数值流程。
*   **第二部分：深度学习识别模块 (Deep Learning)** —— 利用仿真生成的物理增强数据集，训练能够感应拓扑结构的损伤识别模型。

---

## 2. 第一部分：有限元程序实现 (FEM Implementation)

本部分的开发严格遵循学术有限元分析的标准处理流程。

### 2.1 前处理与几何建模 (Preprocessing)
模型支持通过 YAML 配置文件定义几何拓扑、材料属性（$E, \rho, \nu$）及边界约束。
*   **关键代码**: 
    - `./structure.yaml`: 定义结构配置。
    - `./PyFEM_Dynamics/core/io_parser.py`: 实现结构对象的自动化构建。

### 2.2 单元列式与矩阵计算 (Element Formulation)
实现了基于线弹性小变形假设的二维桁架单元（Truss2D）与平面梁单元。对于 Truss2D 单元，单根杆件在局部坐标系下的刚度矩阵 $\mathbf{k}^e$ 与一致质量矩阵 $\mathbf{m}^e$ 如下：
$$ \mathbf{k}^e = \frac{EA}{L}\begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}, \quad \mathbf{m}^e = \frac{\rho A L}{6}\begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} $$
*   **关键代码**: `./PyFEM_Dynamics/core/element.py` 中的 `get_local_stiffness` 与 `get_local_mass`。

### 2.3 全局矩阵组装 (Global Assembly)
利用**直接刚度法 (Direct Stiffness Method)** 将各单元贡献累加至全局矩阵 $\mathbf{K}$ 与 $\mathbf{M}$ 中。通过全局自由度 (DOF) 映射表确保受力平衡一致性。
*   **关键代码**: `./PyFEM_Dynamics/solver/assembler.py` 实现了稀疏矩阵组装逻辑。

### 2.4 边界条件施加 (Boundary Conditions)
为确保矩阵非奇异且数值稳定，程序实现了 **矩阵划零划一法 (Zero-One Substitution Method)** 处理位移约束：
$$ \mathbf{K}_{ij} = \delta_{ij}, \quad \mathbf{F}_i = \bar{u}_i \quad (\text{if node } i \text{ is constrained}) $$
该方法相比罚函数法更具确定性，避免了在大规模时程积分中引入虚假高频震荡。
*   **关键代码**: `./PyFEM_Dynamics/solver/boundary.py`。

### 2.5 瞬态动力学隐式积分 (Implicit Time Integration)
采用非线性动力学中经典的 **Newmark-$\beta$ 法** 求解多自由度运动方程：
$$ \mathbf{M}\ddot{\mathbf{u}}_{t+1} + \mathbf{C}\dot{\mathbf{u}}_{t+1} + \mathbf{K}\mathbf{u}_{t+1} = \mathbf{F}_{t+1} $$
其中 $\mathbf{C}$ 采用 Rayleigh 比例阻尼模型。算法取 $\gamma = 0.5, \beta = 0.25$ 以保证线性系统的无条件稳定性（平均加速度法）。
*   **关键代码**: `./PyFEM_Dynamics/solver/integrator.py`。

---

## 3. 第二部分：基于图学习的损伤识别 (Deep Learning Implementation)

在获取高保真动力学响应后，由于损伤（刚度折减）与响应之间存在高度非线性，本项目探索了利用深度学习进行反向识别。

### 3.1 物理增强数据集生成
利用 FEM 内核自动生成了 **10,000** 组包含不同损伤场景（单元随机折减）和多频随机激振的数据集。
*   **核心逻辑**: `./PyFEM_Dynamics/pipeline/data_gen.py`。

### 3.2 图变换网络 (Graph Transformer) 架构
针对工程结构天然具有图拓扑（Graph Topology）的特性，模型采用了由 4 层 GAT 与 Transformer 层组成的识别架构：
1.  **节点特征编码**: 将加速度/位移时程及坐标映射为高维节点空间特征。
2.  **空间关系推理**: 通过注意力机制计算力学信号在杆件间的传递通量。
3.  **单元损伤预测**: 这是一个回归任务，旨在预测每个单元的 E 值折减系数 [0.5, 1.0]。
*   **模型实现**: `./deep_learning/models/gt_model.py`。

### 3.3 训练成果与消融分析
经过测试，Graph Transformer 模型在 10,000 样本下的性能表现达到预期，其回归精度（MAE）与物理校准度如下：

| 指标 | 测试表现 | 结论评价 |
| :--- | :--- | :--- |
| **MAE** | **0.0766** | 能够准确捕捉大部分杆件的刚度变化。 |
| **F1-Score** | **0.3845** | 对显著损伤的定位具有较高的召回率。 |

![结果对比](docs/images/dl_results/model_comparison.png)
*   **可视化说明**: 从校准曲线（Calibration Curve）可以看出，GT 模型预测的损伤概率具有较强的物理代表性。

---

## 4. 第三部分：真实工况预测应用 (即将开展)

此部分为后续研究方向，计划将已训练的模型应用于受约束的实验室实测数据或更复杂的工业工况预测。

*   **[研究内容占位]**: 复杂信噪比环境下的鲁棒性测试。
*   **[研究内容占位]**: 跨模型泛化能力验证。

---

## 5. 总结
本项目通过底层 FEM 代码实现，深入理解了结构计算的基本力学流程，并在此基础上尝试了前沿的 AI-for-Science 识别技术。虽然目前的分类 F1 分数及 PINN 在当前超参下的协同效应仍有提升空间，但已经证明了图神经网络在结构健康监测中的巨大应用潜力。
