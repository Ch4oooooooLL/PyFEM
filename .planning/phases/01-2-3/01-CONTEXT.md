# Phase 1: 搭建测试框架 - Context

**Gathered:** 2026-03-17
**Status:** Ready for planning

<domain>
## Phase Boundary

建立pytest测试体系，覆盖FEM核心(Node, Element, Material, Section)和DL模型(GT, PINN)，使 `pytest tests/ -v` 全部通过。交付物包括测试基础设施、核心类单元测试、求解器测试、模型测试和集成测试。不添加新功能，仅建立测试框架。

</domain>

<decisions>
## Implementation Decisions

### 测试组织结构
- **目录结构**: 混合组织
  - `tests/fem/` - FEM核心和求解器测试
  - `tests/dl/` - 深度学习模型测试  
  - `tests/integration/` - 集成测试
- **包结构**: 不需要 `__init__.py`，保持tests为普通目录
- **测试文件命名**: `test_<full_path>.py` 格式（如 `test_core_node.py`, `test_solver_assembler.py`）
- **Fixtures位置**: 根目录 `tests/conftest.py`，所有测试共享
- **Pytest配置**: 标准配置，包含路径、markers、过滤和覆盖率设置

### Fixtures设计
- **FEM结构**: 使用简单fixtures，直接在fixture中创建Node/Element对象（不通过YAML加载）
- **PyTorch模型**: 自动检测GPU/CPU，优先GPU，回退CPU
  - 使用 `@pytest.mark.skipif(not torch.cuda.is_available())` 跳过CUDA专用测试
- **临时文件**: 使用pytest内置 `tmp_path` fixture，自动清理
- **作用域**: 默认function scope，每个测试独立fixture

### 测试分类策略
- **慢测试**: 使用 `@pytest.mark.slow` 标记
  - 可通过 `pytest -m "not slow"` 跳过
- **测试类型**: 不刻意区分单元测试和集成测试，统一视为功能测试
- **冒烟测试**: 不单独设立，所有测试都作为冒烟测试运行
- **条件跳过**: 需要 `@pytest.mark.skipif` 装饰器处理环境差异（如无GPU时跳过CUDA测试）

### 覆盖率目标
- **整体目标**: 60%（符合ROADMAP.md要求）
- **豁免代码**:
  - 可视化代码: `postprocess/`, `notebooks/`
  - 主入口脚本和CLI: `main.py`, `run_*.py`
- **报告格式**: HTML报告（`htmlcov/`目录），提供行级覆盖详情

### Claude's Discretion
- pytest.ini的具体配置选项（markers定义、过滤规则）
- fixtures的具体实现细节
- 测试用例的具体断言方式
- 测试数据的具体数值

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Specification
- `.planning/phases/phase-1-testing.md` — Phase 1详细任务分解，测试文件清单，成功标准

### Project Context  
- `.planning/ROADMAP.md` §Phase 1 — 阶段目标和交付物定义
- `.planning/STATE.md` — 项目状态和当前问题

### Codebase Patterns
- `.planning/codebase/STRUCTURE.md` — 代码库结构，模块说明
- `.planning/codebase/CONVENTIONS.md` — 编码规范，命名约定
- `.planning/codebase/TESTING.md` — 现有测试方式（手动冒烟测试）

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **FEM核心类**: `Node`, `TrussElement2D`, `BeamElement2D`, `Material`, `Section` — 需要单元测试
- **求解器类**: `Assembler`, `BoundaryCondition`, `NewmarkBetaSolver` — 需要测试
- **DL模型**: `GTDamagePredictor`, `PINNDamagePredictor`, `PINNLoss` — 需要测试
- **数据管道**: `FEMDataset`, `YAMLParser` — 需要集成测试

### Established Patterns
- **手动测试**: 当前通过运行脚本验证（见TESTING.md）
- **类型注解**: 全代码库使用现代Python类型注解
- **NumPy/PyTorch混合**: FEM用NumPy，DL用PyTorch

### Integration Points
- **pytest**: 需要添加到requirements.txt
- **Conda环境**: 使用 `FEM` 环境
- **目录结构**: 测试目录与PyFEM_Dynamics/, Deep_learning/并列

</code_context>

<specifics>
## Specific Ideas

- 测试应使用现有的结构YAML文件作为测试数据
- 模型测试使用小规模dummy数据（batch_size=2, num_elements=5）确保快速运行
- 可参考现有notebook中的验证逻辑转为测试用例

</specifics>

<deferred>
## Deferred Ideas

- CI/CD集成（GitHub Actions）— 未来阶段
- 性能基准测试 — 未来阶段
- 模糊测试（Fuzzing）— 未来阶段

</deferred>

---

*Phase: 01-2-3*  
*Context gathered: 2026-03-17*
