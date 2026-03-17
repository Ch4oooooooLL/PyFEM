# Phase 2: PINN模型优化 - Context

**Gathered:** 2026-03-17
**Status:** Ready for planning

<domain>
## Phase Boundary

解决PINN模型收敛问题（当前MAE 0.47 vs GT 0.085）。实现自适应损失权重机制，让物理约束与数据项在训练过程中动态平衡。交付物包括自适应损失函数、优化版模型和训练脚本。成功标准：PINN MAE < 0.2。

</domain>

<decisions>
## Implementation Decisions

### 自适应权重策略
- **方法**: Uncertainty Weighting（Kendall et al.）
  - 每个损失项(data/smoothness/range)对应一个可学习log_sigma参数
  - 自动平衡各损失项权重，无需手动调参
- **分阶段训练**:
  - **Warmup阶段**: 使用静态权重让数据损失收敛
  - **切换条件**: 数据损失波动范围（滑动窗口标准差）< 阈值
  - **自适应阶段**: 启用Uncertainty Weighting动态调整
- **优势**: 理论基础扎实，适合PINN多任务场景

### 训练流程设计
- **代码组织**:
  - 新建 `Deep_learning/models/pinn_loss_adaptive.py` - AdaptivePINNLoss类
  - 修改 `Deep_learning/train.py` - 集成新损失函数
- **CLI接口**:
  - `--pinn_mode {static,adaptive}` - 选择训练模式
  - `--use_v2_loss` - 显式启用v2自适应损失
- **Warmup实现**:
  - 单optimizer保持状态连续性
  - 收敛判断：滑动窗口内数据损失标准差 < threshold (默认0.01)
  - 收敛后动态切换PINNLoss → AdaptivePINNLoss
- **向后兼容**: 默认static模式，不破坏现有调用

### 模型架构改进
- **主干网络**: 保持现有3层架构不变
  - 输入维度 → hidden_dim(128) → ReLU → Dropout(0.3)
  - 重复3层 → damage_head/feature_head
- **改进**: 添加轻量级通道注意力(SE-block style)到damage_head
  - Squeeze: 全局平均池化
  - Excitation: 两个FC层 + Sigmoid
  - 参数量增加 < 100，计算开销极小
- **代码组织**:
  - 新建 `Deep_learning/models/attention.py` - SEBlock模块
  - 新建 `Deep_learning/models/pinn_model_v2.py` - 使用SEBlock的改进模型
  - 原有pinn_model.py保持不变用于对比

### 验证与监控
- **监控指标**:
  - 各损失项值: data_loss, smoothness_loss, range_loss, total_physics_loss
  - 自适应权重值: 每个log_sigma的当前值
  - 评估指标: MAE, RMSE, F1, IoU
- **记录方式**:
  - TensorBoard日志: `runs/pinn_experiment_TIMESTAMP/`
  - 使用 `torch.utils.tensorboard.SummaryWriter`
  - 每epoch记录所有指标
- **检查点策略**:
  - 每10 epochs保存: `pinn_checkpoint_epoch_{N}.pth`
  - 验证集最佳模型: `pinn_best.pth`
  - 保存optimizer状态和scheduler状态用于恢复
- **对比实验**:
  - 固定随机种子（42, 123, 456）运行3次
  - 报告指标: 平均值 ± 标准差
  - 对比配置: static vs adaptive两种模式

### Claude's Discretion
- TensorBoard具体布局设计（scalars分组、图表命名）
- 滑动窗口大小（默认5-10个epoch）
- 收敛阈值具体数值（默认0.01）
- SEBlock的具体层维度设计
- 检查点文件命名格式

</decisions>

<specifics>
## Specific Ideas

- 参考Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses"论文实现Uncertainty Weighting
- 使用Phase 1建立的测试框架验证新损失函数
- SE-block设计参考Squeeze-and-Excitation Networks原始论文
- 训练日志应能直观显示权重从warmup到adaptive的切换点

</specifics>

<canonical_refs>
## Canonical References

### Phase Specification
- `.planning/ROADMAP.md` §Phase 2 — PINN优化目标、成功标准、交付物定义
- `.planning/STATE.md` — 当前PINN性能问题和基线指标

### Codebase Context
- `Deep_learning/models/pinn_model.py` — 现有PINNDamagePredictor和PINNLoss实现
- `Deep_learning/train.py` — 现有训练流程，需要集成新损失
- `Deep_learning/models/gt_model.py` — GT模型对比基准（MAE 0.085）

### Prior Phase Context
- `.planning/phases/01-2-3/01-CONTEXT.md` — Phase 1测试框架决策
- `tests/conftest.py` — 测试fixtures和确定性设置

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **PINNLoss** (`pinn_model.py:144-211`): 现有静态损失函数，作为AdaptivePINNLoss的基线
- **PINNDamagePredictor** (`pinn_model.py:8-142`): 完整模型架构，v2版本在此基础上添加SEBlock
- **train.py训练循环** (`train.py:155-376`): 成熟的训练流程，支持AMP、early stopping、scheduler
- **compute_physics_loss** (`pinn_model.py:110-141`): 物理约束计算，AdaptivePINNLoss将复用

### Established Patterns
- **确定性训练**: `train.py:95-122`的`_set_global_determinism`函数，用于对比实验
- **命令行参数**: `train.py:378-402`的参数解析模式，新增参数应遵循相同风格
- **模型保存/加载**: `train.py:362-364`的checkpoint逻辑，需兼容新损失函数状态

### Integration Points
- **train.py修改点**:
  - `train_model()`函数 (line 289): 添加adaptive_mode参数
  - 损失函数初始化 (line 304-312): 支持动态切换PINNLoss/AdaptivePINNLoss
  - 训练循环 (line 331-373): 添加收敛检测和损失函数切换逻辑
- **模型修改点**:
  - `damage_head` (line 51-56): 插入SEBlock模块
  - 保持`forward()`接口不变以兼容现有训练代码

</code_context>

<deferred>
## Deferred Ideas

- 将自适应权重策略迁移到GT模型（Phase 4后评估是否有必要）
- 更复杂的物理约束（如基于应力的物理损失）— 超出当前scope
- 自动超参搜索（如网格搜索lambda值）— 属于自动化调优阶段

</deferred>

---

*Phase: 02-pinn-3-4*  
*Context gathered: 2026-03-17*
