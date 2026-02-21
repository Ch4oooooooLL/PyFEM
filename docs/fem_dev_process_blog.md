# PyFEM-Dynamics 开发记录（按有限元求解步骤）

这篇文档按照有限元实际计算的顺序，整理了我在 `PyFEM-Dynamics` 中的实现过程。每一步都包含三部分：
- 这一步的基本公式
- 程序里的数值处理方式
- Python 关键代码片段

目标不是写成论文，而是把“从公式到代码”的路径讲清楚。


---

## 0. 项目做了什么

这个项目主要完成了三件事：
1. 2D 桁架/梁结构静力分析
2. 2D 结构瞬态动力分析（Newmark-beta）
3. 批量生成损伤数据（用于后续机器学习）

核心目录：

```text
PyFEM_Dynamics/
├── core/        # 节点、单元、材料、截面、输入解析
├── solver/      # 组装、边界条件、静力/动力求解
├── pipeline/    # 批量样本生成
├── postprocess/ # 画图
└── main.py      # 单次静力测试入口
```

---

## 1. 第一步：离散模型（节点和单元）

### 1.1 基本公式

离散后，静力和动力方程分别是：

$$
\mathbf{K}\mathbf{U}=\mathbf{F}
$$

$$
\mathbf{M}\ddot{\mathbf{U}}+\mathbf{C}\dot{\mathbf{U}}+\mathbf{K}\mathbf{U}=\mathbf{F}(t)
$$

### 1.2 数值处理

程序中把对象拆成：
- `Node`：坐标和自由度编号
- `Element`：单元矩阵和坐标变换
- `Material`、`Section`：物理参数

### 1.3 关键代码

```python
# core/node.py
class Node:
    def __init__(self, node_id: int, x: float, y: float):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.dofs = []
```

```python
# core/io_parser.py
has_beam = any(isinstance(e, BeamElement2D) for e in elements)
dofs_per_node = 3 if has_beam else 2
```

---

## 2. 第二步：材料和截面参数

### 2.1 基本公式

模型里最关键的参数是：
- 弹性模量 $E$
- 密度 $\rho$
- 截面面积 $A$
- 惯性矩 $I$

例如，桁架轴向刚度是 $\frac{EA}{L}$，梁弯曲刚度与 $EI$ 有关。

### 2.2 数值处理

材料和截面单独做成类，后面做参数变化和损伤模拟会更方便。

### 2.3 关键代码

```python
# core/material.py
class Material:
    def __init__(self, E: float, rho: float):
        self.E = E
        self.rho = rho
```

```python
# core/section.py
class Section:
    def __init__(self, A: float, I: float = 0.0):
        self.A = A
        self.I = I
```

---

## 3. 第三步：单元刚度矩阵和质量矩阵

## 3.1 Truss2D 单元

### 公式

桁架单元只考虑轴向：

$$
k=\frac{EA}{L}
$$

### 数值处理

质量矩阵支持两种：
- 一致质量（更标准）
- 集中质量（更快）

### 关键代码

```python
k = E * A / L
return np.array([
    [ k, 0, -k, 0],
    [ 0, 0,  0, 0],
    [-k, 0,  k, 0],
    [ 0, 0,  0, 0]
], dtype=float)
```

```python
if lumping:
    m = m_total / 2.0
    return np.diag([m, m, m, m])
```

## 3.2 Beam2D 单元（Euler-Bernoulli）

### 公式

梁单元每个节点有 3 个自由度（$u,v,\theta$），弯曲部分常用系数：

$$
\frac{12EI}{L^3},\;\frac{6EI}{L^2},\;\frac{4EI}{L},\;\frac{2EI}{L}
$$

### 数值处理

按标准 $6\times 6$ 梁单元模板实现。

### 关键代码

```python
k_v1 = 12 * E * I / (L**3)
k_v2 = 6 * E * I / (L**2)
k_v3 = 4 * E * I / L
k_v4 = 2 * E * I / L
```

---

## 4. 第四步：坐标变换（局部到全局）

### 4.1 基本公式

$$
\mathbf{K}^e_g = \mathbf{T}^\mathsf{T}\mathbf{K}^e_l\mathbf{T},\quad
\mathbf{M}^e_g = \mathbf{T}^\mathsf{T}\mathbf{M}^e_l\mathbf{T}
$$

### 4.2 数值处理

先算单元方向角，再构造变换矩阵 $\mathbf{T}$。

### 4.3 关键代码

```python
c = np.cos(self.angle)
s = np.sin(self.angle)
return T.T @ k_local @ T
```

---

## 5. 第五步：组装全局矩阵

### 5.1 基本公式

$$
\mathbf{K}=\sum_e \mathbf{L}_e^\mathsf{T}\mathbf{K}^e\mathbf{L}_e,
\quad
\mathbf{M}=\sum_e \mathbf{L}_e^\mathsf{T}\mathbf{M}^e\mathbf{L}_e
$$

### 5.2 数值处理

- 组装用 `lil_matrix`
- 求解前转 `csc_matrix`

### 5.3 关键代码

```python
K_global = sp.lil_matrix((self.total_dofs, self.total_dofs), dtype=float)
for element in self.elements:
    dofs = element.node1.dofs + element.node2.dofs
    K_global[np.ix_(dofs, dofs)] += element.get_global_stiffness()
return K_global.tocsc()
```

---

## 6. 第六步：载荷向量构造

### 6.1 基本公式

静力是 $\mathbf{F}$，动力是 $\mathbf{F}(t)$。

### 6.2 数值处理

- `LOAD`：直接加到对应自由度
- `DLOAD`：按时间区间写到荷载矩阵对应列

### 6.3 关键代码

```python
# 静力
F[gdof] += val
```

```python
# 动力
start_step = int(t_start / dt)
end_step = int(t_end / dt)
global_F[gdof, start_step:min(end_step, num_steps)] = val
```

---

## 7. 第七步：边界条件处理

### 7.1 基本思路

位移边界条件如果不处理，会出现奇异矩阵。项目实现了两种常见方法：
- 划零划一法
- 罚函数法（乘大数）

### 7.2 数值处理

- 静力：常用划零划一
- 批量动力：常用 penalty，流程更直接

### 7.3 关键代码

```python
self.dirichlet_bcs.append((dof_index, value))
```

```python
# 罚函数法
K_mod[dof, dof] *= penalty
F_mod[dof] = K_mod[dof, dof] * val
```

```python
# 划零划一
K_lil.rows[dof] = [dof]
K_lil.data[dof] = [1.0]
F_mod[dof] = val
```

---

## 8. 第八步：静力求解与内力回算

### 8.1 基本公式

$$
\mathbf{K}_{mod}\mathbf{U}=\mathbf{F}_{mod}
$$

### 8.2 数值处理

用 `spsolve` 解位移，再回算单元轴力。

### 8.3 关键代码

```python
U = spla.spsolve(K, F)
```

```python
u_el = np.array([
    U[el.node1.dofs[0]], U[el.node1.dofs[1]],
    U[el.node2.dofs[0]], U[el.node2.dofs[1]]
])
f_local = k_local @ (T @ u_el)
axial_force = f_local[2]
```

---

## 9. 第九步：动力时程积分（Newmark-beta）

### 9.1 基本公式

阻尼矩阵：

$$
\mathbf{C}=\alpha\mathbf{M}+\beta\mathbf{K}
$$

每步等效刚度：

$$
\hat{\mathbf{K}} = \mathbf{K}+a_0\mathbf{M}+a_1\mathbf{C}
$$

### 9.2 数值处理

- 先算初始加速度
- $\hat{\mathbf{K}}$ 只分解一次（LU）
- 每步回代更新位移、速度、加速度

### 9.3 关键代码

```python
a0 = 1.0 / (self.beta * self.dt**2)
a1 = self.gamma / (self.beta * self.dt)
K_hat = self.K + a0 * self.M + a1 * self.C
K_hat_lu = spla.splu(K_hat.tocsc())
```

```python
F_hat = F_t[:, i] + self.M.dot(term_M) + self.C.dot(term_C)
u_next = K_hat_lu.solve(F_hat)
a_next = a0 * (u_next - u_prev) - a2 * v_prev - a3 * a_prev
v_next = v_prev + a6 * a_prev + a7 * a_next
```

---

## 10. 第十步：批量损伤样本生成

### 10.1 基本思路

把损伤简化为弹性模量退化：

$$
E_d = \eta E_0,\quad \eta\in[0.5, 0.9]
$$

每个样本随机损伤 $1\text{--}3$ 个单元，然后做完整动力分析。

### 10.2 数值处理

- 每个样本重新加载模型，避免相互影响
- 只提取传感器节点加速度
- 保存为 `sensor_X.npy` 和 `damage_Y.npy`

### 10.3 关键代码

```python
num_damaged = random.randint(1, 3)
damaged_indices = random.sample(range(len(elements)), num_damaged)
for idx in damaged_indices:
    factor = random.uniform(0.5, 0.9)
    damage_factors[idx] = factor
    elements[idx].material.E *= factor
```

```python
sensor_data = A[sensor_dofs, :] if sensor_dofs else A
X_list.append(sensor_data)
Y_list.append(y_label)
np.save(os.path.join(save_dir, "sensor_X.npy"), np.array(X_list))
np.save(os.path.join(save_dir, "damage_Y.npy"), np.array(Y_list))
```

---

## 11. 小结

这个项目目前已经实现了一个完整链路：
- 有限元建模
- 静力/动力求解
- 损伤数据生成

后续可以继续做：
1. 增加更多单元类型（如 Timoshenko 梁、平面应力单元）
2. 增加模态分析和频响分析
3. 做参数识别和不确定性分析

整体上，这个项目对我最大的意义是把“公式-离散-求解-数据输出”完整跑通，为后续更复杂的结构分析打下基础。
