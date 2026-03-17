# Roadmap

## Goal
解决FEM-DL代码库的4个关键问题：测试缺失、PINN模型性能、输入验证、性能瓶颈

## Phase 1: 搭建测试框架 (2-3天)
**问题**: 无测试框架，靠手动冒烟测试
**目标**: 建立pytest测试体系，覆盖FEM核心和DL模型
**交付物**:
- `tests/conftest.py` - pytest配置和fixtures
- `tests/test_fem_core/` - Node, Element, Assembler单元测试
- `tests/test_solver/` - Boundary, Integrator测试
- `tests/test_models/` - GT, PINN模型测试
- `tests/test_pipeline/` - 集成测试
**成功标准**: `pytest tests/ -v` 全部通过

## Phase 2: PINN模型优化 (3-4天)
**问题**: PINN MAE 0.47，100 epochs仅提升0.28%
**根因**: 物理约束与数据项量级不匹配，静态权重
**目标**: 实现自适应损失权重，提升PINN收敛
**交付物**:
- `Deep_learning/models/pinn_loss.py` - 自适应权重损失函数
- `Deep_learning/models/pinn_model_v2.py` - 优化版模型
- `Deep_learning/train_pinn_v2.py` - 支持动态权重训练
**成功标准**: PINN MAE < 0.2 (从0.47降低)

## Phase 3: YAML配置验证 (2天)
**问题**: 配置无schema验证，可能导致运行时错误
**目标**: 添加Pydantic模型验证所有YAML配置
**交付物**:
- `PyFEM_Dynamics/config/schemas.py` - Pydantic schema定义
- `PyFEM_Dynamics/config/validator.py` - 验证器
- 修改 `io_parser.py` - 集成验证
**成功标准**: 无效配置提前报错，提供清晰错误信息

## Phase 4: 数据生成并行化 (2-3天)
**问题**: 20,000样本需30分钟，单线程
**目标**: 使用multiprocessing加速数据生成
**交付物**:
- `PyFEM_Dynamics/pipeline/data_gen_parallel.py` - 并行版本
- `PyFEM_Dynamics/pipeline/worker.py` - 工作进程
- 保留原接口，添加 `--parallel` 和 `--n-jobs` 参数
**成功标准**: 4核加速3-3.5倍，20,000样本<10分钟

## 依赖关系
```
Phase 1 (Testing)
    ↓
Phase 2 (PINN) ──→ Phase 4 (Performance)
    ↓
Phase 3 (Validation)
```

## 总时间估计
- 开发: 9-12天
- 测试验证: +2天
- **总计: 11-14天**

## 环境要求
- **Conda Environment**: `FEM`
- **Dependencies to add**: pytest, pydantic
