# Phase 3: YAML配置验证 - Context

**Gathered:** 2026-03-17
**Status:** Ready for planning

<domain>
## Phase Boundary

添加Pydantic模型验证所有YAML配置文件，确保无效配置在加载时就被捕获并提供清晰的错误信息。交付物包括schema定义文件、验证器模块和io_parser集成。仅验证用户编辑的配置文件，内部模板可跳过。不添加新功能，仅增强配置验证能力。

</domain>

<decisions>
## Implementation Decisions

### 验证范围
- **验证目标**: 仅验证用户会编辑的YAML配置文件
  - `structure.yaml` — 结构定义（节点、单元、材料、边界条件）
  - `dataset_config.yaml` — 数据集生成配置
- **跳过验证**: `load_template.yaml`（内部模板，非用户配置）、`condition_case.yaml`（可后续按需添加）
- **优先级**: `structure.yaml` 最核心，需完整验证；`dataset_config.yaml` 次之

### 验证模式
- **字段验证**: 严格模式 — 拒绝任何未知字段，报错提示"Unexpected field: xxx"
- **数值验证**: 合理性范围验证 — 验证数值在物理合理范围内
  - 正数检查: E > 0, dt > 0, A > 0
  - 比例检查: 0 < damage_factor < 1
  - 范围检查: min < max, t_start < t_end
- **类型验证**: 完整类型检查（int/float/str/list/dict）

### 错误处理
- **语言**: 中英双语 — 主要信息中文，技术细节英文
  - 例: "结构定义错误: nodes[2].coords 期望 [float, float]，实际得到3个值"
- **详细程度**: 详细模式 — 显示完整路径、错误原因、建议修复方法
  - 包含YAML文件路径、字段层级、期望类型/值、实际类型/值
- **错误格式**: 结构化错误信息，便于程序化处理和人工阅读

### 验证时机
- **触发方式**: 加载时自动验证 — 修改 `io_parser.py`，在 `yaml.safe_load()` 后立即验证
- **验证函数**: 每个配置类型有独立的 `validate_*()` 函数
- **失败处理**: 立即抛出 `ValidationError`，包含所有验证错误，不逐条报错
- **调用示例**:
  ```python
  from config.validator import validate_structure
  data = validate_structure(yaml.safe_load(f))
  ```

### Claude's Discretion
- Pydantic模型具体字段定义和类型选择
- 数值范围的精确边界值（如E的最小值设多少）
- 错误信息的具体措辞和格式
- 验证器与io_parser的集成细节
- 是否添加自定义验证装饰器

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Specification
- `.planning/ROADMAP.md` §Phase 3 — 阶段目标、交付物、成功标准
- `.planning/phases/phase-3-config-validation.md` — Phase 3详细任务分解（如有）

### Project Context
- `.planning/STATE.md` — 项目状态和当前问题
- `.planning/phases/01-2-3/01-CONTEXT.md` — Phase 1决策（pytest使用、类型注解要求）

### Codebase Patterns
- `.planning/codebase/STRUCTURE.md` — 代码库结构，模块说明
- `.planning/codebase/CONVENTIONS.md` — 编码规范，命名约定

### Existing Code
- `PyFEM_Dynamics/core/io_parser.py` — 现有YAML解析器，需要集成验证
- `structure.yaml` — 结构配置示例
- `dataset_config.yaml` — 数据集配置示例

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **YAMLParser类** (`io_parser.py`): 现有解析逻辑可作为验证后的处理流程
- **StructureData dataclass**: 可作为Pydantic模型的参考结构
- **现有YAML文件**: `structure.yaml` 和 `dataset_config.yaml` 可作为测试用例

### Established Patterns
- **类型注解**: 全代码库使用现代Python类型注解，Pydantic模型需保持一致
- **错误处理**: 使用 `ValueError` 抛出具体错误信息（如"Element {eid} references undefined material"）
- **dataclass**: 现有 `StructureData` 使用 `@dataclass`，Pydantic的BaseModel类似但更强大

### Integration Points
- **io_parser.py**: 验证器需与现有解析器集成，在 `load_structure_yaml()` 和 `load_dataset_config()` 中调用
- **pytest框架**: Phase 1建立的测试框架，验证器需要配套单元测试
- **requirements.txt**: 需要添加 `pydantic` 依赖

### Config Structure
**structure.yaml schema要点**:
- metadata: description, num_nodes, num_elements, num_dofs, dofs_per_node
- materials: list of {name, E, rho, nu}
- nodes: list of {id, coords: [x, y]}
- elements: list of {id, nodes: [n1, n2], material, A, I}
- boundary: list of {node_id, constraints: [ux, uy, rz]}

**dataset_config.yaml schema要点**:
- structure_file, output_file: str paths
- time: {dt, total_time}
- generation: {num_samples, random_seed}
- damage: {enabled, min_damaged_elements, max_damaged_elements, reduction_range}
- load_generation: complex nested structure with modes, ranges, weights
- damping: {alpha, beta}
- deep_learning: {train: {...}, inference: {...}}

</code_context>

<specifics>
## Specific Ideas

- 错误信息示例: `"structure.yaml → elements[5].material: 引用未定义材料 'steel2'，可用材料: ['steel']"`
- 验证应能捕获常见错误: 节点ID不连续、材料未定义、坐标数量错误、时间范围倒置
- 考虑使用Pydantic的 `field_validator` 进行自定义验证（如检查节点ID唯一性）
- 数值范围参考: E ∈ [1e9, 1e12], dt ∈ [1e-6, 1.0], damage_factor ∈ [0.0, 1.0]

</specifics>

<deferred>
## Deferred Ideas

- `condition_case.yaml` 验证 — 可后续按需添加
- `load_template.yaml` 验证 — 内部模板，非用户配置
- 配置文件自动补全/IDE支持（JSON Schema）— 未来阶段
- 配置版本迁移工具（schema变更时自动转换旧配置）— 未来阶段
- Web界面配置编辑器 — 未来阶段

</deferred>

---

*Phase: 03-yaml-validation*  
*Context gathered: 2026-03-17*
