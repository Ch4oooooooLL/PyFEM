# PyFEM-Dynamics 二维结构动力学有限元程序

## 1. 简介
本程序是一个使用 Python 编写的二维有限元求解器，主要用于学习计算桁架和梁结构的静力与动力响应。本程序是为了数据驱动的结构健康监测研究提供基于物理模型的仿真计算结果。程序主要包含结构有限元建模、静/动力学求解以及结果后处理可视化模块。

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
`pipeline/data_gen.py` 模块提供了批量循环计算的功能。程序可以通过对选定单元的弹性模量 $E$ 进行折减（例如乘以 $0.5 \sim 0.9$ 的系数），来模拟部分杆件发生刚度退化的损伤工况。随后对每种工况进行动力时程分析，提取指定节点（模拟传感器位置）的加速度等响应信号，最终保存为特征数据集。

## 3. 使用方法与输入格式
程序支持从文本文件（`.txt` 和 `.csv`）中读取结构模型与分析参数。关于节点坐标、单元拓扑、边界条件和外力载荷等详细指令的书写规范，请参考项目中的 `INPUT_FORMAT.md` 说明文件。项目根目录下的 `static_input.txt` 和 `batch_input.txt` 包含了典型的 Warren 桥梁桁架结构示例供参考。

## 4. 算例与结果展示

程序内置了绘图接口，支持对输出的节点位移、单元力以及时间历史进行可视化评估：

**1. 静力分析演示：桁架受载变形与轴力分布**
模型计算后，利用内置方法通过不同颜色深浅或色系隐式映射了每个单元内部的轴引力或压力，并可以在图形上将小变形按照给定缩放因子进行放大，直观呈现节点变位：
![static_deformation](docs/images/static_deformation.png)

**2. 动力参数化抽样：传感器信号与损伤分布**
针对数据生成管线的动态计算抽取样本，能够将输入的随机退化损伤矩阵同时间步积分响应以组合图形式同步展现。以底部某监测节点的双向加速度为例：
![ai4m_sample](docs/images/ai4m_sample_0.png)

