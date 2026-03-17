# Phase 3 Research: YAML Configuration Validation

**Phase:** 03-yaml-validation  
**Date:** 2026-03-17  
**Status:** Complete

---

## 1. Existing YAML Structure Analysis

### 1.1 structure.yaml

**Schema Overview:**
```yaml
metadata:
  description: str          # 结构描述
  num_nodes: int            # 节点数量
  num_elements: int         # 单元数量
  num_dofs: int             # 总自由度
  dofs_per_node: int        # 每节点自由度 (2 for truss, 3 for beam)

materials:                  # 材料列表
  - name: str               # 材料名称
    E: float                # 弹性模量 (Pa), must be > 0
    rho: float              # 密度 (kg/m³), optional, default 0.0
    nu: float               # 泊松比, optional, default 0.3

nodes:                      # 节点列表
  - id: int                 # 节点ID (0-indexed)
    coords: [float, float]  # [x, y] 坐标

elements:                   # 单元列表
  - id: int                 # 单元ID (0-indexed)
    nodes: [int, int]       # [node1_id, node2_id]
    material: str           # 材料名称引用
    A: float                # 截面积 (m²), must be > 0
    I: float                # 惯性矩, optional, default 0.0

boundary:                   # 边界条件列表 (optional)
  - node_id: int            # 约束节点ID
    constraints: [str]      # ["ux", "uy", "rz"] 的子集
```

**Validation Requirements:**
- `num_nodes` must match nodes list length
- `num_elements` must match elements list length
- Node IDs must be unique and sequential (0 to N-1)
- Element node references must exist
- Material names must be defined in materials section
- Material references in elements must be valid
- Constraints must be valid dof strings
- Positive values: E, A, I (when > 0)

### 1.2 dataset_config.yaml

**Schema Overview:**
```yaml
structure_file: str         # 结构文件路径
output_file: str            # 输出文件路径

time:
  dt: float                 # 时间步长 (s), > 0
  total_time: float         # 总时间 (s), > dt

generation:
  num_samples: int          # 样本数, >= 1
  random_seed: int          # 随机种子

damage:
  enabled: bool             # 是否启用
  min_damaged_elements: int # 最少损伤单元
  max_damaged_elements: int # 最多损伤单元
  reduction_range: [float, float]  # [min, max], 0 < min < max < 1

load_generation:
  mode: str                 # "random_multi_point" | "static" | "harmonic"
  random_multi_point:       # 仅当 mode == "random_multi_point"
    num_loads_per_sample_range: [int, int]  # [min, max]
    candidate_nodes:
      mode: str             # "all_non_support_nodes" | "include" | "exclude"
      include_nodes: [int]  # 可选
      exclude_nodes: [int]  # 可选
    dof_candidates: [str]   # ["fx", "fy", "mz"] 子集
    dof_weights: [float]    # 权重列表, sum ≈ 1.0
    pattern_candidates: [str]  # ["harmonic", "pulse", "half_sine", "gaussian", "ramp"]
    pattern_weights: [float]
    avoid_duplicate_node_dof: bool
    parameter_ranges:
      F0: [float, float]    # 力幅值范围
      freq: [float, float]  # 频率范围, > 0
      phase: [float, float] # 相位范围
      offset: [float, float] # 偏置范围
      t_start: [float, float]  # 开始时间
      t_end: [float, float]    # 结束时间, > t_start
      t_ramp: [float, float]   # 斜坡时间, > 0
      t0: [float, float]       # 高斯中心
      sigma: [float, float]    # 高斯宽度, > 0

damping:
  alpha: float              # 质量阻尼系数
  beta: float               # 刚度阻尼系数

deep_learning:
  train:
    model: str              # "gt" | "pinn" | "both"
    dataset_mode: str       # "response" | "load" | "both"
    epochs: int             # >= 1
    batch_size: int         # >= 1
    lr: float               # 学习率, > 0
    hidden_dim: int         # 隐藏层维度, >= 1
    lambda_smooth: float    # 平滑损失权重, >= 0
    threshold: float        # 损伤阈值, 0 < threshold < 1
    seed: int
    deterministic: bool
    deterministic_warn_only: bool
    num_threads: int
    device: str             # "auto" | "cuda" | "cpu"
    amp: bool
    num_workers: int
    pin_memory: bool
  inference:
    model_type: str         # "gt" | "pinn"
    feature_mode: str       # "response" | "load" | "both"
    checkpoint: str         # 模型路径
    dataset_npz: str        # 数据集路径
    device: str             # "auto" | "cuda" | "cpu"
```

**Validation Requirements:**
- Range validations: min < max for all [min, max] pairs
- Time consistency: t_end > t_start, total_time > dt
- Enum validations: mode, model, dataset_mode, device, feature_mode
- Positive values: dt, num_samples, epochs, batch_size, lr, hidden_dim
- Probability values: threshold ∈ (0, 1), reduction_range ∈ (0, 1)
- Array length consistency: dof_candidates and dof_weights same length
- File paths exist (optional, maybe warning only)

---

## 2. Pydantic Model Design

### 2.1 Recommended Model Hierarchy

```python
# schemas.py structure

# --- Base Models ---
class FEMBaseModel(BaseModel):
    """Base model with strict validation"""
    model_config = ConfigDict(
        extra='forbid',  # Reject unknown fields
        validate_assignment=True,
        str_strip_whitespace=True,
    )

# --- structure.yaml Models ---
class Metadata(FEMBaseModel):
    description: str
    num_nodes: int = Field(gt=0)
    num_elements: int = Field(gt=0)
    num_dofs: int = Field(gt=0)
    dofs_per_node: int = Field(ge=2, le=3)

class Material(FEMBaseModel):
    name: str = Field(min_length=1)
    E: float = Field(gt=0, le=1e15)  # Elastic modulus
    rho: float = Field(ge=0, default=0.0)  # Density
    nu: float = Field(ge=-1.0, le=0.5, default=0.3)  # Poisson's ratio

class Node(FEMBaseModel):
    id: int = Field(ge=0)
    coords: List[float] = Field(min_length=2, max_length=2)

class Element(FEMBaseModel):
    id: int = Field(ge=0)
    nodes: List[int] = Field(min_length=2, max_length=2)
    material: str = Field(min_length=1)
    A: float = Field(gt=0)  # Cross-sectional area
    I: float = Field(ge=0, default=0.0)  # Moment of inertia

class BoundaryCondition(FEMBaseModel):
    node_id: int = Field(ge=0)
    constraints: List[str]  # Validated in model_validator

class StructureConfig(FEMBaseModel):
    """Root model for structure.yaml"""
    metadata: Metadata
    materials: List[Material] = Field(min_length=1)
    nodes: List[Node] = Field(min_length=2)
    elements: List[Element] = Field(min_length=1)
    boundary: List[BoundaryCondition] = Field(default_factory=list)
    
    @model_validator(mode='after')
    def validate_consistency(self):
        # Check counts match
        # Check node ID uniqueness and sequential
        # Check element references
        # Check material references
        # Check boundary node references
        return self

# --- dataset_config.yaml Models ---
class TimeConfig(FEMBaseModel):
    dt: float = Field(gt=0, le=10.0)
    total_time: float = Field(gt=0, le=1e6)
    
    @model_validator(mode='after')
    def validate_time(self):
        if self.total_time <= self.dt:
            raise ValueError(f'total_time ({self.total_time}) must be > dt ({self.dt})')
        return self

class GenerationConfig(FEMBaseModel):
    num_samples: int = Field(ge=1, le=1_000_000)
    random_seed: int

class DamageConfig(FEMBaseModel):
    enabled: bool = True
    min_damaged_elements: int = Field(ge=0)
    max_damaged_elements: int = Field(ge=0)
    reduction_range: Tuple[float, float]
    
    @model_validator(mode='after')
    def validate_ranges(self):
        if self.max_damaged_elements < self.min_damaged_elements:
            raise ValueError('max_damaged_elements < min_damaged_elements')
        min_red, max_red = self.reduction_range
        if not (0 < min_red < max_red < 1):
            raise ValueError('reduction_range must be (min, max) with 0 < min < max < 1')
        return self

# ... (additional models for load_generation, damping, deep_learning)

class DatasetConfig(FEMBaseModel):
    """Root model for dataset_config.yaml"""
    structure_file: str
    output_file: str
    time: TimeConfig
    generation: GenerationConfig
    damage: DamageConfig
    load_generation: LoadGenerationConfig
    damping: DampingConfig
    deep_learning: DeepLearningConfig
```

### 2.2 Key Design Decisions

**Strict Mode (extra='forbid'):**
- Prevents typos in field names from being silently ignored
- Ensures schema evolves intentionally

**Field Validators:**
- Use Pydantic's `Field()` for simple constraints (gt, ge, lt, le)
- Use `field_validator` for cross-field dependencies
- Use `model_validator` for complex consistency checks

**Type Annotations:**
- Use modern Python types: `list[int]`, `tuple[float, float]`
- Optional fields use `Field(default=...)` or `| None`

---

## 3. Custom Validators Required

### 3.1 Structure Config Validators

```python
@model_validator(mode='after')
def validate_node_ids(self):
    """Check node IDs are unique and sequential"""
    ids = [n.id for n in self.nodes]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate node IDs found')
    expected = set(range(self.metadata.num_nodes))
    actual = set(ids)
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        raise ValueError(f'Node IDs mismatch. Missing: {missing}, Extra: {extra}')
    return self

@model_validator(mode='after')
def validate_element_references(self):
    """Check element nodes exist and materials are defined"""
    node_ids = {n.id for n in self.nodes}
    mat_names = {m.name for m in self.materials}
    
    for elem in self.elements:
        for nid in elem.nodes:
            if nid not in node_ids:
                raise ValueError(f'Element {elem.id} references undefined node {nid}')
        if elem.material not in mat_names:
            available = ', '.join(sorted(mat_names))
            raise ValueError(f'Element {elem.id} references undefined material "{elem.material}". Available: [{available}]')
    return self

@field_validator('constraints')
@classmethod
def validate_constraints(cls, v):
    """Check boundary constraints are valid"""
    valid = {'ux', 'uy', 'rz'}
    invalid = set(v) - valid
    if invalid:
        raise ValueError(f'Invalid constraints: {invalid}. Valid: {valid}')
    return v
```

### 3.2 Dataset Config Validators

```python
@field_validator('mode')
@classmethod
def validate_load_mode(cls, v):
    valid_modes = {'random_multi_point', 'static', 'harmonic'}
    if v not in valid_modes:
        raise ValueError(f'Invalid load mode "{v}". Must be one of: {valid_modes}')
    return v

@model_validator(mode='after')
def validate_array_lengths(self):
    """Check dof_candidates and dof_weights match"""
    if len(self.dof_candidates) != len(self.dof_weights):
        raise ValueError(f'dof_candidates ({len(self.dof_candidates)}) and dof_weights ({len(self.dof_weights)}) must have same length')
    return self
```

---

## 4. Integration with io_parser.py

### 4.1 Current Flow
```python
# Current (io_parser.py)
with open(file_path, 'r') as f:
    data = yaml.safe_load(f)
# Manual validation scattered throughout...
```

### 4.2 New Flow
```python
# New flow with Pydantic
from PyFEM_Dynamics.config.schemas import StructureConfig, DatasetConfig
from PyFEM_Dynamics.config.validator import ValidationError

def load_structure_yaml(file_path: str) -> StructureData:
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = yaml.safe_load(f)
    
    # Validate with Pydantic
    try:
        validated = StructureConfig.model_validate(raw_data)
    except ValidationError as e:
        raise ValueError(format_validation_errors(e, file_path)) from e
    
    # Continue with existing conversion logic...
    return convert_to_structure_data(validated)

def load_dataset_config(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = yaml.safe_load(f)
    
    try:
        validated = DatasetConfig.model_validate(raw_data)
    except ValidationError as e:
        raise ValueError(format_validation_errors(e, file_path)) from e
    
    return validated.model_dump()
```

### 4.3 Error Message Formatter

```python
def format_validation_errors(error: ValidationError, file_path: str) -> str:
    """Convert Pydantic errors to bilingual, user-friendly messages"""
    messages = []
    for err in error.errors():
        loc = ' → '.join(str(x) for x in err['loc'])
        msg = err['msg']
        
        # Map to Chinese context
        messages.append(f"配置文件错误 [{file_path}]\n  位置: {loc}\n  原因: {msg}")
    
    return '\n\n'.join(messages)
```

---

## 5. Testing Strategy

### 5.1 Unit Tests for Schemas

```python
# tests/test_config/test_schemas.py

class TestStructureConfig:
    def test_valid_structure(self):
        data = load_yaml_fixture('valid_structure.yaml')
        config = StructureConfig.model_validate(data)
        assert config.metadata.num_nodes == 7
    
    def test_duplicate_node_ids(self):
        data = load_yaml_fixture('duplicate_nodes.yaml')
        with pytest.raises(ValidationError) as exc:
            StructureConfig.model_validate(data)
        assert 'Duplicate node IDs' in str(exc.value)
    
    def test_undefined_material(self):
        data = load_yaml_fixture('undefined_material.yaml')
        with pytest.raises(ValidationError) as exc:
            StructureConfig.model_validate(data)
        assert 'undefined material' in str(exc.value)
    
    def test_extra_fields_rejected(self):
        data = {'metadata': {...}, 'extra_field': 'value'}
        with pytest.raises(ValidationError):
            StructureConfig.model_validate(data)
```

### 5.2 Integration Tests for Validator

```python
# tests/test_config/test_validator.py

class TestValidatorIntegration:
    def test_validate_structure_file(self, tmp_path):
        yaml_file = tmp_path / 'structure.yaml'
        yaml_file.write_text(valid_structure_yaml)
        
        result = validate_structure_file(str(yaml_file))
        assert isinstance(result, StructureConfig)
    
    def test_invalid_yaml_syntax(self, tmp_path):
        yaml_file = tmp_path / 'bad.yaml'
        yaml_file.write_text('invalid: yaml: content: [')
        
        with pytest.raises(YAMLError):
            validate_structure_file(str(yaml_file))
```

### 5.3 Test Fixtures Needed

- `valid_structure.yaml` - Base valid config
- `duplicate_nodes.yaml` - Duplicate node IDs
- `undefined_material.yaml` - Element references non-existent material
- `missing_field.yaml` - Required field missing
- `extra_field.yaml` - Unknown field present
- `invalid_range.yaml` - Values out of valid range
- `valid_dataset_config.yaml` - Valid dataset config
- `inconsistent_time.yaml` - total_time < dt

---

## 6. Implementation Plan

### 6.1 File Structure

```
PyFEM_Dynamics/
└── config/
    ├── __init__.py
    ├── schemas.py          # Pydantic models
    ├── validator.py        # Validation functions
    └── errors.py           # Custom error classes
```

### 6.2 Dependencies

Add to `requirements.txt`:
```
pydantic>=2.0.0
```

### 6.3 Implementation Order

1. **Create config package** - `__init__.py`, `errors.py`
2. **Implement StructureConfig schemas** - Start with core models
3. **Add structure.yaml validators** - Node/element consistency checks
4. **Integrate into io_parser.py** - Replace manual validation
5. **Implement DatasetConfig schemas** - More complex nested models
6. **Add dataset_config.yaml validators** - Range and enum checks
7. **Write comprehensive tests** - Unit and integration tests

### 6.4 Migration Path

- No breaking changes to existing YAML files
- Invalid configs will now fail fast with clear errors
- Existing valid configs will continue to work
- Add `--skip-validation` flag for emergency bypass (optional)

---

## 7. Technical Considerations

### 7.1 Performance

- Pydantic v2 is fast (Rust core), validation overhead is minimal
- Validation happens once at load time, not during computation
- No impact on FEM solving performance

### 7.2 Error Messages

- Use Chinese for user-facing messages
- Include technical details (field types, expected values)
- Show exact file path and line location (via YAML loader)
- Group multiple errors together (Pydantic default behavior)

### 7.3 Extensibility

- Easy to add new config types (e.g., `condition_case.yaml`)
- Schema versioning can be added later if needed
- Custom validators can be reused across models

### 7.4 IDE Support

- Pydantic models enable autocomplete in modern IDEs
- Consider generating JSON Schema for advanced IDE features (future)

---

## 8. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Overly strict validation breaks existing configs | Test against all existing YAML files before deployment |
| Poor error messages confuse users | Iterate on message format with example errors |
| Performance regression | Benchmark validation time on large configs |
| Circular imports with io_parser | Keep schemas in separate package, use forward refs |

---

## 9. Success Criteria

Per ROADMAP.md Phase 3 requirements:
- ✅ `PyFEM_Dynamics/config/schemas.py` - Pydantic schema definitions
- ✅ `PyFEM_Dynamics/config/validator.py` - Validation functions
- ✅ Modified `io_parser.py` - Integrated validation
- ✅ Invalid configs fail fast with clear bilingual errors
- ✅ All existing tests pass
- ✅ New validation tests added

---

*End of Research Document*
