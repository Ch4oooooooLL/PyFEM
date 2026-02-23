# 二维结构动力学仿真与损伤识别研究实践

## 1. 项目简介
本项目涉及结构有限元方法和数据驱动损伤识别的编程实践。项目共分为两个主要模块：
*   **有限元计算内核** —— 基于 Python 定制开发的二维有限元内核，支持静力分析及基于 Newmark-beta 方法的动力响应仿真。
*   **深度学习识别模块** —— 探索利用图变换网络（Graph Transformer）感应物理拓扑，进而对结构单元损伤进行预测。

---

## 2. 第一部分：有限元程序实现流程

有限元分析内核严格按照标准的数值计算流程进行组织，核心算法实现如下：

### 2.1 建模与参数输入
程序支持通过扩展性较强的 YAML 格式定义几何拓扑。相关解析逻辑位于 `./PyFEM_Dynamics/core/io_parser.py`。

### 2.2 单元刚度与质量矩阵计算
针对二维拉压桁架单元，其在局部坐标系下的矩阵计算逻辑如下。
*   **源码位置**: `./PyFEM_Dynamics/core/element.py`

```python
# 局部刚度矩阵计算片段
def get_local_stiffness(self):
    E, A, L = self.material.E, self.section.A, self.length
    k = E * A / L
    return np.array([
        [ k,  0, -k,  0],
        [ 0,  0,  0,  0],
        [-k,  0,  k,  0],
        [ 0,  0,  0,  0]
    ])
```

### 2.3 全局矩阵组装
利用直接刚度法，将各单元贡献累加至全局矩阵。为了优化大规模计算下的内存表现，采用了稀疏矩阵存储。
*   **源码位置**: `./PyFEM_Dynamics/solver/assembler.py`

```python
# 全局刚度矩阵组装片段
def assemble_K(self):
    K_global = sp.lil_matrix((self.total_dofs, self.total_dofs))
    for element in self.elements:
        k_global_element = element.get_global_stiffness()
        dofs = element.node1.dofs + element.node2.dofs
        K_global[np.ix_(dofs, dofs)] += k_global_element
    return K_global.tocsc()
```

### 2.4 边界条件的数值处理
程序实现了**划零划一法**来施加本质边界条件。该算法通过修改受约束自由度对应的行列，确保数值解的稳定性。
*   **源码位置**: `./PyFEM_Dynamics/solver/boundary.py`

```python
# 划零划一法核心逻辑片段
for dof in bc_dofs:
    # 列清零 (CSC格式)
    K_csc.data[K_csc.indptr[dof]:K_csc.indptr[dof+1]] = 0.0
# 行清零并设置对角线为1 (LIL格式)
for dof, val in self.dirichlet_bcs:
    K_lil.rows[dof] = [dof]
    K_lil.data[dof] = [1.0]
    F_mod[dof] = val
```

### 2.5 动力学时间积分求解
动力响应计算采用了隐式 **Newmark-$\beta$ 法**。
*   **源码位置**: `./PyFEM_Dynamics/solver/integrator.py`

```python
# Newmark-beta 时间步迭代循环关键部分
for i in range(1, self.num_steps):
    # 计算当前步等效荷载 F_hat
    F_hat = F_t[:, i] + self.M.dot(term_M) + self.C.dot(term_C)
    # 求解 U_i (利用预分解的 K_hat_lu)
    u_next = K_hat_lu.solve(F_hat)
    # 更新加速度 a_next 和速度 v_next
    a_next = a0 * (u_next - u_prev) - a2 * v_prev - a3 * a_prev
    v_next = v_prev + a6 * a_prev + a7 * a_next
```

---

## 3. 第二部分：基于图学习的损伤识别研究

在获取动力学响应数据后，利用深度学习模型对构件状态进行逆向评估。

### 3.1 训练数据准备
通过 `./PyFEM_Dynamics/pipeline/data_gen.py` 自动化生成了 **10,000** 组样本。每组样本包含结构在特定激振下的节点响应时程。

### 3.2 深度学习模型架构
模型选取了能够直接处理非结构化网格拓扑的 **Graph Transformer**。
*   **源码位置**: `./deep_learning/models/gt_model.py`

```python
# GT模型核心交互层片段
class GTDamagePredictor(nn.Module):
    def forward(self, x, adj, edge_index):
        # 1. 节点时间特征提取 (Conv1d)
        h_node = self.node_encoder(x_flat).squeeze(-1)
        # 2. 空间特征交互 (GAT Layers)
        h_node = self.gat1(h_node, adj)
        h_node = self.gat2(h_node, adj)
        # 3. 边关系聚合与损伤预测 (MLP)
        damage = self.damage_fc(torch.cat([h_node1, h_node2], dim=-1))
        return damage
```

### 3.3 训练成果分析
模型在独立测试集上的表现如下表所示：

| 评估指标 | 指标数值 | 简要评价 |
| :--- | :--- | :--- |
| **平均绝对误差 (MAE)** | **0.076** | 预测值与真实刚度折减系数较为接近。 |
| **F1-Score** | **0.384** | 具备初步识别显著损伤位置的能力。 |

![训练历史](docs/images/dl_results/gt_history.png)
*   **说明**: 训练曲线表明，针对特定拓扑结构，模型可以从动力学信号中归纳出基础的受力规律。

---

## 4. 第三部分：后续研究方向 (占位)

本项目预留了以下待补充的研究章节：

*   **[子项占位]**: 复杂信噪比环境下的抗干扰识别测试。
*   **[子项占位]**: 跨拓扑结构的预训练模型迁移学习探索。

---

## 5. 小结
本项目完成了从底层力学内核开发到高阶深度学习应用的闭环实践。通过直接查阅源码中的数值算法实现，系统地验证了仿真驱动的 SHM（结构健康监测）方法论的可行性。
