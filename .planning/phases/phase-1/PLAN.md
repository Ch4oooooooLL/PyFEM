# Phase 1: 搭建测试框架 (Testing Framework Setup)

## Goal
建立pytest测试体系，覆盖FEM核心、求解器和DL模型，解决当前无测试框架的问题。

## Success Criteria
- [ ] `pytest tests/ -v` 运行通过，无报错
- [ ] 核心FEM类(Node, Element, Assembler)有完整单元测试
- [ ] 求解器组件(Boundary, Integrator)有单元测试
- [ ] DL模型(GT, PINN)有基础功能测试
- [ ] 测试覆盖率 ≥ 60%
- [ ] 集成测试验证YAML解析和数据生成流程

## Tasks

### Task 1: Setup pytest infrastructure
**Description**: 初始化pytest测试框架基础配置

**Files to Modify**:
- `requirements.txt` - 添加pytest>=7.0.0, pytest-cov>=4.0.0

**Files to Create**:
- `tests/__init__.py`
- `tests/conftest.py` - pytest fixtures和配置
- `pytest.ini` - pytest运行配置

**Verification**:
```bash
pip install -r requirements.txt
pytest --version  # 应显示pytest版本
```

---

### Task 2: FEM Core Unit Tests
**Description**: 为核心FEM类编写单元测试

**Files to Create**:
- `tests/test_fem_core/__init__.py`
- `tests/test_fem_core/test_node.py`
  - Node坐标存储测试
  - DOF分配测试
  - 边界条件标记测试
- `tests/test_fem_core/test_element.py`
  - TrussElement2D长度计算测试
  - 角度计算测试
  - 局部刚度矩阵测试
  - 坐标变换矩阵测试
- `tests/test_fem_core/test_material.py`
  - Material属性访问测试
- `tests/test_fem_core/test_section.py`
  - Section面积和惯性矩测试

**Verification**:
```bash
pytest tests/test_fem_core/ -v
```

---

### Task 3: Solver Unit Tests
**Description**: 为求解器组件编写单元测试

**Files to Create**:
- `tests/test_solver/__init__.py`
- `tests/test_solver/test_assembler.py`
  - 全局刚度矩阵K组装测试
  - 全局质量矩阵M组装测试
  - 稀疏矩阵格式测试
- `tests/test_solver/test_boundary.py`
  - 零一法边界条件应用测试
  - 多点约束测试
- `tests/test_solver/test_integrator.py`
  - Newmark-β常数计算测试
  - 时间步进测试

**Verification**:
```bash
pytest tests/test_solver/ -v
```

---

### Task 4: DL Model Tests
**Description**: 为深度学习模型编写基础功能测试

**Files to Create**:
- `tests/test_models/__init__.py`
- `tests/test_models/test_gt_model.py`
  - GTDamagePredictor实例化测试
  - 前向传播输入输出形状测试
  - 模型参数数量测试
- `tests/test_models/test_pinn_model.py`
  - PINNDamagePredictor实例化测试
  - 前向传播输入输出形状测试
  - 物理损失计算测试

**Verification**:
```bash
pytest tests/test_models/ -v
```

---

### Task 5: Data Pipeline Tests
**Description**: 为数据管道组件编写测试

**Files to Create**:
- `tests/test_pipeline/__init__.py`
- `tests/test_pipeline/test_dataset.py`
  - FEMDataset加载测试
  - 数据形状验证
  - 归一化测试
- `tests/test_pipeline/test_yaml_parser.py`
  - YAML解析roundtrip测试
  - 配置验证测试

**Verification**:
```bash
pytest tests/test_pipeline/ -v
```

---

### Task 6: Integration Tests
**Description**: 编写端到端集成测试

**Files to Create**:
- `tests/test_integration/__init__.py`
- `tests/test_integration/test_fem_pipeline.py`
  - 从YAML到FEM模拟的完整流程
  - 小数据集生成测试(10 samples)
- `tests/test_integration/test_training_pipeline.py`
  - 训练流程集成测试(1 epoch)

**Verification**:
```bash
pytest tests/test_integration/ -v -m "not slow"
```

---

### Task 7: Coverage and Configuration
**Description**: 配置测试覆盖率和运行环境

**Files to Modify**:
- `pytest.ini` - 添加覆盖率配置和markers
- `.gitignore` - 排除测试产物

**Files to Create**:
- `.coveragerc` - 覆盖率配置

**Verification**:
```bash
pytest tests/ --cov=. --cov-report=html --cov-report=term
# 验证覆盖率 ≥ 60%
```

---

### Task 8: Test Fixtures and Utilities
**Description**: 创建共享的测试夹具和工具函数

**Files to Modify**:
- `tests/conftest.py`
  - 添加Node fixtures
  - 添加Element fixtures
  - 添加Material fixtures
  - 添加临时目录fixtures
  - 添加模拟数据fixtures

**Verification**:
```bash
pytest tests/ --fixtures  # 验证fixtures已注册
```

---

### Task 9: Smoke Test Suite
**Description**: 将原有手动冒烟测试转换为自动化测试

**Files to Create**:
- `tests/test_smoke/__init__.py`
- `tests/test_smoke/test_entry_points.py`
  - main.py可导入测试
  - train.py可导入测试
  - data_gen.py可导入测试

**Verification**:
```bash
pytest tests/test_smoke/ -v
```

---

### Task 10: Final Verification
**Description**: 完整测试套件运行和修复

**Verification**:
```bash
# 完整测试运行
pytest tests/ -v --tb=short

# 覆盖率检查
pytest tests/ --cov=. --cov-report=term-missing

# 慢测试标记检查
pytest tests/ -m slow -v
```

---

## Dependencies

```
Task 1 (Setup)
    ↓
Task 8 (Fixtures)
    ↓
Task 2 (FEM Core) ──┐
Task 3 (Solver) ────┼──→ Task 6 (Integration)
Task 4 (DL Models) ─┤
Task 5 (Pipeline) ──┘
    ↓
Task 9 (Smoke Tests)
    ↓
Task 7 (Coverage)
    ↓
Task 10 (Final Verification)
```

## Task Allocation Summary

| Task | Est. Time | Files Created | Complexity |
|------|-----------|---------------|------------|
| 1. Setup | 30 min | 3 | Low |
| 2. FEM Core | 2 hours | 5 | Medium |
| 3. Solver | 2 hours | 4 | Medium |
| 4. DL Models | 1.5 hours | 3 | Medium |
| 5. Pipeline | 1 hour | 3 | Low |
| 6. Integration | 1.5 hours | 3 | Medium |
| 7. Coverage | 30 min | 1 | Low |
| 8. Fixtures | 1 hour | - (mod) | Medium |
| 9. Smoke Tests | 30 min | 2 | Low |
| 10. Final | 1 hour | - | Low |

**Total Est. Time**: ~11 hours

## Notes

- 使用 `tmp_path` fixture避免真实文件系统操作
- 标记慢测试: `@pytest.mark.slow`
- 标记需要GPU的测试: `@pytest.mark.gpu`
- 使用fixtures共享测试数据
- Conda环境: FEM
- Python版本: 3.9+
