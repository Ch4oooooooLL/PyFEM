# STATE

## Project
- **Name**: FEM-DL Codebase Improvement
- **Status**: Planning
- **Phase**: 0/4
- **Conda Environment**: FEM

## Codebase Context
- **Type**: Python FEM + Deep Learning
- **Lines of Code**: ~3,100 Python
- **Current Issues**: No tests, PINN underperforms, no validation, performance bottleneck

## Problems to Solve
1. **Testing** (High) - No formal test framework
2. **PINN Model** (High) - Poor convergence (MAE 0.47 vs GT 0.085)
3. **Input Validation** (Medium) - No YAML schema validation
4. **Performance** (Medium) - Data generation single-threaded

## Environment
```bash
conda activate FEM
```

## Phases
- [x] Phase 1: Testing Framework (pytest) - **COMPLETE** → `.planning/phases/01-testing-framework/`
- [ ] Phase 2: PINN Loss Weight Optimization
- [ ] Phase 3: Configuration Validation - **CONTEXT CAPTURED** → `.planning/phases/03-yaml-validation/`
- [ ] Phase 4: Parallel Data Generation

## Current Phase
**Phase 3** - YAML Configuration Validation
- **Status**: Context captured, ready for planning
- **Location**: `.planning/phases/03-yaml-validation/03-CONTEXT.md`
- **Context gathered**: 2026-03-17
- **Key decisions**:
  - 仅验证用户配置: structure.yaml, dataset_config.yaml
  - 严格模式: 拒绝未知字段
  - 详细错误: 中英双语，显示完整路径
  - 自动验证: 加载时自动检查

## Completed Phases
- [x] Phase 1: Testing Framework — 74 tests, pytest + coverage configured

## Commands
```bash
# Test
pytest tests/ -v

# Train
python Deep_learning/train.py --model gt --epochs 1

# Validate config
python -c "from config_validator import validate; validate('structure.yaml')"

# Generate data (optimized)
python PyFEM_Dynamics/pipeline/data_gen.py --parallel
```
