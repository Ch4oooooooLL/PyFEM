# Phase 2: PINN模型优化

## Goal
解决PINN收敛问题，MAE从0.47降至<0.2

## Root Cause Analysis
- 物理约束损失与数据损失量级不匹配
- 静态权重 `lambda_phys=0.1` 不调整
- 训练过程中约束冲突未解决

## Success Criteria
- [ ] PINN MAE < 0.2 (当前0.47)
- [ ] 训练收敛稳定，无明显震荡
- [ ] 相对提升 > 10% (当前仅0.28%)

## Approach: Adaptive Loss Weighting

### 2.1 实现GradNorm-style自适应权重
参考论文: "GradNorm: Gradient Normalization for Adaptive Loss Balancing"

```python
class AdaptivePINNLoss(nn.Module):
    """
    自适应权重损失:
    - 数据项: ||pred - true||
    - 物理平滑项: ||Δpred||
    - 范围约束项: range penalty
    
    权重动态调整基于:
    - 各损失项的相对大小
    - 各损失项的梯度范数
    """
```

### 2.2 实现Uncertainty Weighting
参考Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses"

为每个损失项学习一个不确定性参数:
```python
log_sigma_data = nn.Parameter(torch.zeros(1))
log_sigma_smooth = nn.Parameter(torch.zeros(1))
log_sigma_range = nn.Parameter(torch.zeros(1))

loss = 0.5 * (loss_data / sigma_data**2 + log_sigma_data) + \
       0.5 * (loss_smooth / sigma_smooth**2 + log_sigma_smooth) + ...
```

## Tasks

### 2.1 Create adaptive loss module
**New file**: `Deep_learning/models/pinn_loss.py`
- `GradNormLoss` - 梯度归一化版本
- `UncertaintyWeightedLoss` - 不确定性加权版本
- `StaticWeightedLoss` - 原始版本（baseline）

### 2.2 Update PINN model
**Modify**: `Deep_learning/models/pinn_model.py`
- 支持切换不同损失函数
- 保留向后兼容

### 2.3 Create optimized training script
**New file**: `Deep_learning/train_pinn_adaptive.py`
- 支持 `--loss-type` 参数
- 添加权重历史记录和可视化
- 对比实验: static vs gradnorm vs uncertainty

### 2.4 Evaluation
**New file**: `Deep_learning/eval_pinn.py`
- 加载训练好的模型
- 计算MAE, RMSE, F1
- 生成对比图表

## Experiments

| Loss Type | Expected MAE | Config |
|-----------|--------------|--------|
| Static | 0.45-0.50 | lambda=[1.0, 0.1, 0.1] |
| GradNorm | 0.15-0.20 | alpha=1.5 |
| Uncertainty | 0.15-0.22 | learned |

## Notes
- 从GT模型复制架构作为baseline
- 训练epochs: 50-100 (验证收敛性)
- 保留checkpoint对比
- **Conda环境**: FEM
