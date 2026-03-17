# Phase 3: YAML配置验证

## Goal
添加schema验证，防止无效配置导致运行时错误

## Success Criteria
- [ ] 所有YAML文件有Pydantic schema
- [ ] 无效配置在启动时报错
- [ ] 错误信息清晰（指出具体字段和问题）
- [ ] 向后兼容现有配置文件

## Schemas to Define

### 3.1 Structure Schema
```python
class Node(BaseModel):
    id: int = Field(ge=0)
    coords: Tuple[float, float]

class Element(BaseModel):
    id: int = Field(ge=0)
    nodes: Tuple[int, int]
    material: str
    A: float = Field(gt=0)
    I: float = Field(ge=0)

class StructureConfig(BaseModel):
    metadata: Metadata
    materials: List[Material]
    nodes: List[Node]
    elements: List[Element]
    boundary: List[BoundaryCondition]
```

### 3.2 Dataset Config Schema
```python
class TimeConfig(BaseModel):
    dt: float = Field(gt=0)
    total_time: float = Field(gt=0)

class DamageConfig(BaseModel):
    enabled: bool
    min_damaged_elements: int = Field(ge=1)
    max_damaged_elements: int
    reduction_range: Tuple[float, float]

class DatasetConfig(BaseModel):
    structure_file: FilePath
    output_file: str
    time: TimeConfig
    generation: GenerationConfig
    damage: DamageConfig
    load_generation: LoadGenerationConfig
    damping: DampingConfig
    deep_learning: DeepLearningConfig
```

### 3.3 Condition Case Schema
```python
class ConditionCase(BaseModel):
    structure_file: FilePath
    time: TimeConfig
    damping: DampingConfig
    load_case: LoadCase
    damage_case: DamageCase
    inference: InferenceConfig
    output_dir: str = "outputs"
```

## Tasks

### 3.1 Create schemas module
**New directory**: `PyFEM_Dynamics/config/`
**New file**: `PyFEM_Dynamics/config/schemas.py`
- Pydantic models for all configs
- Custom validators for cross-field checks

### 3.2 Create validator
**New file**: `PyFEM_Dynamics/config/validator.py`
```python
def validate_structure(path: str) -> StructureConfig:
    """验证并解析structure.yaml"""
    
def validate_dataset_config(path: str) -> DatasetConfig:
    """验证并解析dataset_config.yaml"""

def validate_condition_case(path: str) -> ConditionCase:
    """验证并解析condition_case.yaml"""
```

### 3.3 Integrate into parsers
**Modify**: `PyFEM_Dynamics/core/io_parser.py`
- 使用validator验证YAML
- 保持原有接口

### 3.4 Add CLI validation tool
**New file**: `scripts/validate_config.py`
```bash
python scripts/validate_config.py structure.yaml
python scripts/validate_config.py dataset_config.yaml
python scripts/validate_config.py condition_case.yaml
```

## Validation Rules

### Structure
- [ ] 节点ID唯一
- [ ] 单元引用的节点存在
- [ ] 材料名称被引用
- [ ] 边界条件节点存在
- [ ] A > 0, I >= 0

### Dataset Config
- [ ] structure_file存在
- [ ] min_damaged <= max_damaged
- [ ] reduction_range在[0,1]内
- [ ] dt * total_time 合理 (<10,000步)

### Condition Case
- [ ] 引用的checkpoint文件存在（如果是eval）
- [ ] output_dir可写

## Error Messages
```
ValidationError: structure.yaml
  - nodes[0].coords: Field required
  - elements[5].nodes: Node 99 does not exist
  - boundary[1].node_id: Node 6 has no constraints
```

## Notes
- Pydantic v2 语法
- 使用 `FilePath` 类型验证文件存在
- **Conda环境**: FEM
- 添加依赖: `pydantic>=2.0.0`
