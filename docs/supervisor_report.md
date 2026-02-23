# PyFEM-Dynamics 工作记录

## 1. 有限元程序生成数据

### 1.1 建模思路

本项目先用 2D 桁架有限元做动力学仿真，再批量制造损伤样本，最终生成深度学习训练数据。

控制方程：

$$
\mathbf{M}\ddot{\mathbf{u}}(t)+\mathbf{C}\dot{\mathbf{u}}(t)+\mathbf{K}\mathbf{u}(t)=\mathbf{F}(t)
$$

Rayleigh 阻尼：

$$
\mathbf{C}=\alpha\mathbf{M}+\beta\mathbf{K}
$$

Newmark-$\beta$（平均加速度）：

$$
\hat{\mathbf{K}}=\mathbf{K}+a_0\mathbf{M}+a_1\mathbf{C},
\quad
a_0=\frac{1}{\beta\Delta t^2},
\quad
a_1=\frac{\gamma}{\beta\Delta t}
$$

损伤模拟（对单元弹性模量折减）：

$$
E_i^{(d)}=d_iE_i,\quad d_i\in[0.5,0.9]
$$

---

### 1.2 当前数据集配置与规模

- 时间步长：`dt = 0.01 s`
- 总时长：`2.0 s`
- 样本数：`2000`
- 结构规模：`7` 节点，`11` 单元，`14` 自由度

数据文件：`dataset/train.npz`  
数据键值与形状：

- `load`: `(2000, 201, 14)`
- `E`: `(2000, 11)`
- `disp`: `(2000, 201, 14)`
- `stress`: `(2000, 201, 11)`
- `damage`: `(2000, 11)`

---

### 1.3 关键算法代码（精简）

#### 1.3.1 批量样本生成主循环

```python
for i in range(num_samples):
    nodes_i, elements_i, bcs_i = build_structure_from_yaml(structure_file)

    damage_vec, E_damaged = apply_damage(elements_i, damage_config, rng)
    load_matrix = generate_load_matrix(load_specs, num_nodes, 2, dt, num_steps, rng)
    disp, stress = run_fem_solver(nodes_i, elements_i, bcs_i, load_matrix, dt, total_time, alpha, beta)

    all_loads[i] = load_matrix
    all_E[i] = E_damaged
    all_disp[i] = disp
    all_stress[i] = stress
    all_damage[i] = damage_vec
```

#### 1.3.2 Newmark 时间积分核心更新

```python
# K_hat = K + a0*M + a1*C
K_hat = self.K + a0 * self.M + a1 * self.C
K_hat_lu = spla.splu(K_hat.tocsc())

for i in range(1, self.num_steps):
    term_M = a0 * u_prev + a2 * v_prev + a3 * a_prev
    term_C = a1 * u_prev + a4 * v_prev + a5 * a_prev
    F_hat = F_t[:, i] + self.M.dot(term_M) + self.C.dot(term_C)

    u_next = K_hat_lu.solve(F_hat)
    a_next = a0 * (u_next - u_prev) - a2 * v_prev - a3 * a_prev
    v_next = v_prev + a6 * a_prev + a7 * a_next
```

---

### 1.4 图示

静力变形（用于检查边界条件和受力是否合理）：

![Static deformation](images/supervisor/static_deformation.png)

动力应力云图抽帧（用于检查动态响应空间分布）：

![VM cloud frame](images/supervisor/vm_cloud_frame_100.png)

该图重生成命令：

```bash
python PyFEM_Dynamics/postprocess/generate_vm_cloud.py --sample-id 0 --frame-step 10 --representative-frame 100
```

---

### 1.5 详细代码路径（相对路径）

- `PyFEM_Dynamics/pipeline/data_gen.py`
- `PyFEM_Dynamics/solver/integrator.py`
- `PyFEM_Dynamics/solver/assembler.py`
- `PyFEM_Dynamics/solver/boundary.py`
- `PyFEM_Dynamics/solver/stress_recovery.py`
- `PyFEM_Dynamics/postprocess/generate_vm_cloud.py`
- `dataset_config.yaml`
- `structure.yaml`
- `dataset/metadata.json`

---

## 2. 深度学习部分

### 2.1 任务定义

输入是时序响应（当前训练默认用位移响应），输出是每个单元的损伤系数：

$$
\hat{\mathbf{d}}=f_\theta(\mathbf{X}),\quad \hat{\mathbf{d}}\in[0,1]^{11}
$$

分类判定阈值为 $\tau=0.95$：

$$
\hat{y}_i=\mathbb{I}(\hat{d}_i<\tau)
$$

---

### 2.2 两个模型

| 模型 | 核心思想 |
|---|---|
| LSTM | 直接学习时序响应与损伤之间的映射，侧重数据拟合能力 |
| PINN | 在数据损失外增加物理约束项，鼓励预测更“物理一致” |

---

### 2.3 关键算法代码（精简）

#### 2.3.1 LSTM 前向核心

```python
lstm_out, _ = self.lstm(x)
attn_weights = self.attention(lstm_out)
context = torch.sum(attn_weights * lstm_out, dim=1)
damage = self.fc(context)
return damage
```

对应形式：

$$
\mathbf{h}_t=\mathrm{BiLSTM}(\mathbf{x}_t),\ 
\mathbf{c}=\sum_t \alpha_t \mathbf{h}_t,\ 
\hat{\mathbf{d}}=\sigma(\mathrm{MLP}(\mathbf{c}))
$$

#### 2.3.2 PINN 物理约束核心

```python
diff = damage.unsqueeze(-1) - damage.unsqueeze(-2)
smooth_loss = (adj * diff ** 2).mean()
range_loss = torch.mean(torch.relu(damage - 1.0) ** 2 + torch.relu(-damage) ** 2)
total_physics_loss = lambda_smooth * smooth_loss + lambda_range * range_loss
```

对应形式：

$$
\mathcal{L}_{smooth}=\frac{1}{N}\sum_{i,j}A_{ij}(d_i-d_j)^2,\quad
\mathcal{L}_{range}=\frac{1}{N}\sum_i\left[\max(d_i-1,0)^2+\max(-d_i,0)^2\right]
$$

#### 2.3.3 训练与评估核心

```python
pred = model(x)
loss = criterion(pred, y)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

metrics = compute_metrics(all_preds, all_labels, threshold=threshold)
```

---

### 2.4 详细代码路径（相对路径）

- `Deep_learning/data/dataset.py`
- `Deep_learning/models/lstm_model.py`
- `Deep_learning/models/pinn_model.py`
- `Deep_learning/train.py`
- `Deep_learning/utils/metrics.py`
- `Deep_learning/utils/visualization.py`

---

## 3. 深度学习结果解读

### 3.1 训练曲线

LSTM：

![LSTM history](images/supervisor/lstm_history.png)

PINN：

![PINN history](images/supervisor/pinn_history.png)

### 3.2 结果理解（入门角度）

1. LSTM 的损失下降更快、最终误差更低，说明它更擅长“回归具体损伤值”。
2. PINN 的 `Recall` 很高，说明它“宁可多报，不愿漏报”。
3. PINN 的 `Precision` 和 `Accuracy` 偏低，说明误报比较多，后续需要调损失权重和阈值。

### 3.3 指标计算公式

回归误差：

$$
\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^N|\hat{d}_i-d_i|,
\quad
\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^N(\hat{d}_i-d_i)^2}
$$

分类指标：

$$
\mathrm{Precision}=\frac{TP}{TP+FP},\quad
\mathrm{Recall}=\frac{TP}{TP+FN},\quad
F1=\frac{2PR}{P+R}
$$

---

## 4. 两个模型训练结果对比

### 4.1 单次测试集结果

| 模型 | MAE | RMSE | Accuracy | Precision | Recall | F1 | IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 0.0876 | 0.1277 | 0.5167 | 0.1940 | 0.5024 | 0.2799 | 0.1627 |
| PINN | 0.4708 | 0.4774 | 0.1870 | 0.1870 | 1.0000 | 0.3150 | 0.1870 |

### 4.2 跨运行稳定性（2 次）

- LSTM：`MAE = 0.0842 ± 0.0066`（95% CI），`F1 = 0.2606 ± 0.0379`
- PINN：`MAE = 0.4743 ± 0.0068`（95% CI），`F1 = 0.2918 ± 0.0455`

稳定性图：

![Run stability](images/supervisor/run_stability.png)

综合对比图：

![Model comparison](images/supervisor/model_comparison.png)

### 4.3 小结

1. 如果目标是更准确估计损伤程度，当前 LSTM 更合适。
2. 如果目标是尽量不漏检，当前 PINN 的召回率更有优势。
3. 实际应用中可以考虑“两阶段”：PINN 粗筛 + LSTM 精估。

---

## 5. 复现实验命令

```bash
python PyFEM_Dynamics/pipeline/data_gen.py
python Deep_learning/train.py --model both --epochs 100 --threshold 0.95
python Deep_learning/utils/visualization.py --checkpoints_dir Deep_learning/checkpoints --aggregate_all
```
