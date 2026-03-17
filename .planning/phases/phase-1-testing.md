# Phase 1: 搭建测试框架

## Goal
建立pytest测试体系，解决无测试问题

## Success Criteria
- [ ] `pytest tests/ -v` 运行通过
- [ ] 核心FEM类(Node, Element, Assembler)有单元测试
- [ ] 求解器(Boundary, Integrator)有测试
- [ ] DL模型(GT, PINN)有基础测试
- [ ] 测试覆盖率 > 60%

## Tasks

### 1.1 Setup pytest infrastructure
```bash
# Add to requirements.txt
pytest>=7.0.0
pytest-cov>=4.0.0
```

**Files to modify**:
- `requirements.txt` - 添加pytest依赖

**Files to create**:
- `tests/conftest.py` - pytest配置
- `tests/__init__.py`
- `pytest.ini` - pytest配置

### 1.2 FEM Core Tests
```python
# tests/test_fem_core/test_node.py
# tests/test_fem_core/test_element.py  
# tests/test_fem_core/test_material.py
# tests/test_fem_core/test_section.py
```

**Test cases**:
- Node: coordinate storage, DOF assignment
- TrussElement2D: length/angle calc, stiffness matrix
- Material: property access
- Section: area/inertia

### 1.3 Solver Tests
```python
# tests/test_solver/test_assembler.py
# tests/test_solver/test_boundary.py
# tests/test_solver/test_integrator.py
```

**Test cases**:
- Assembler: K/M matrix assembly, sparse format
- Boundary: zero-one method correctness
- Integrator: Newmark constants, time stepping

### 1.4 DL Model Tests
```python
# tests/test_models/test_gt_model.py
# tests/test_models/test_pinn_model.py
```

**Test cases**:
- Model instantiation
- Forward pass with dummy data
- Output shape validation

### 1.5 Integration Tests
```python
# tests/test_pipeline/test_yaml_parser.py
# tests/test_pipeline/test_data_gen.py
```

**Test cases**:
- YAML parsing roundtrip
- Small dataset generation (10 samples)

## Notes
- 使用 fixtures 共享测试数据
- 避免测试真实文件系统，使用 tmp_path
- 标记慢测试: `@pytest.mark.slow`
- **Conda环境**: FEM
