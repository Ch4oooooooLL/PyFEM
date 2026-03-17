# Phase 3: YAML Configuration Validation - Execution Plan

**Phase:** 03-yaml-validation  
**Status:** Ready for Execution  
**Estimated Duration:** 2 days  
**Dependencies:** Phase 1 (Testing Framework) ✓ COMPLETE

---

## Phase Goal

Add Pydantic-based validation to all user-editable YAML configuration files, ensuring invalid configurations are caught at load time with clear, bilingual error messages.

---

## Success Criteria

- [ ] `PyFEM_Dynamics/config/schemas.py` - Pydantic schema definitions for structure.yaml and dataset_config.yaml
- [ ] `PyFEM_Dynamics/config/validator.py` - Validation functions and error formatters
- [ ] `PyFEM_Dynamics/config/errors.py` - Custom validation error classes
- [ ] `PyFEM_Dynamics/config/__init__.py` - Package exports
- [ ] Modified `PyFEM_Dynamics/core/io_parser.py` - Integrated validation calls
- [ ] Updated `requirements.txt` - Added `pydantic>=2.0.0`
- [ ] All existing YAML files pass validation
- [ ] Invalid configs produce clear, bilingual error messages
- [ ] Unit tests for all schema validators
- [ ] Integration tests for io_parser validation
- [ ] pytest passes with new tests

---

## Task Breakdown

### Task 1: Create Config Package Structure
**Priority:** P0 (Blocking)  
**Duration:** 15 min  
**Dependencies:** None

Create the config package directory structure:
```
PyFEM_Dynamics/config/
├── __init__.py
├── schemas.py
├── validator.py
└── errors.py
```

**Implementation:**
- Create `PyFEM_Dynamics/config/` directory
- Create `__init__.py` with package exports
- Create empty `schemas.py`, `validator.py`, `errors.py`

**Verification:**
```bash
python -c "from PyFEM_Dynamics.config import schemas; print('OK')"
```

---

### Task 2: Define Error Classes
**Priority:** P0 (Blocking)  
**Duration:** 20 min  
**Dependencies:** Task 1

Create custom error classes in `errors.py` for validation failures.

**Implementation:**
```python
# errors.py
class ConfigValidationError(ValueError):
    """Base validation error with bilingual message support"""
    def __init__(self, message_zh: str, message_en: str = "", details: dict = None):
        self.message_zh = message_zh
        self.message_en = message_en
        self.details = details or {}
        super().__init__(f"{message_zh}\n{message_en}")

class StructureValidationError(ConfigValidationError):
    """Structure.yaml validation error"""
    pass

class DatasetConfigValidationError(ConfigValidationError):
    """Dataset_config.yaml validation error"""
    pass
```

**Verification:**
```bash
python -c "from PyFEM_Dynamics.config.errors import ConfigValidationError; raise ConfigValidationError('测试', 'test')"
# Should show bilingual error
```

---

### Task 3: Implement Structure.yaml Base Schemas
**Priority:** P0 (Blocking)  
**Duration:** 45 min  
**Dependencies:** Task 2

Implement Pydantic models for structure.yaml components.

**Implementation in `schemas.py`:**
1. `FEMBaseModel` - Base model with strict validation
2. `Metadata` - Metadata fields with validation
3. `Material` - Material definition
4. `Node` - Node coordinates
5. `Element` - Element definition
6. `BoundaryCondition` - Boundary constraints
7. `StructureConfig` - Root model (without complex validators yet)

**Key Validations:**
- `extra='forbid'` - Reject unknown fields
- `E > 0`, `A > 0`, `I >= 0` - Physical constraints
- `dofs_per_node` in [2, 3] - Valid DOF counts
- Constraints in valid set `{'ux', 'uy', 'rz'}`

**Verification:**
```python
from PyFEM_Dynamics.config.schemas import StructureConfig
import yaml

with open('structure.yaml') as f:
    data = yaml.safe_load(f)
    
config = StructureConfig.model_validate(data)
print(f"✓ Valid: {config.metadata.description}")
```

---

### Task 4: Add Structure.yaml Cross-Field Validators
**Priority:** P0  
**Duration:** 60 min  
**Dependencies:** Task 3

Add complex validators that check consistency across fields.

**Implementation:**
Add `@model_validator(mode='after')` methods to `StructureConfig`:

1. `validate_node_ids()` - Check unique and sequential IDs
2. `validate_element_references()` - Check nodes and materials exist
3. `validate_boundary_references()` - Check boundary node IDs exist
4. `validate_metadata_counts()` - Check counts match list lengths

**Error Messages:**
- Chinese: "结构定义错误: elements[5].material 引用未定义材料 'steel2'，可用材料: ['steel']"
- Include file path, field location, expected vs actual

**Verification:**
```python
# Test with invalid config
invalid_data = {
    'metadata': {'description': 'Test', 'num_nodes': 2, 'num_elements': 1, 'num_dofs': 4, 'dofs_per_node': 2},
    'materials': [{'name': 'steel', 'E': 2e11, 'rho': 7850, 'nu': 0.3}],
    'nodes': [{'id': 0, 'coords': [0, 0]}, {'id': 1, 'coords': [1, 0]}],
    'elements': [{'id': 0, 'nodes': [0, 2], 'material': 'steel', 'A': 0.01}],
}

from pydantic import ValidationError
try:
    StructureConfig.model_validate(invalid_data)
except ValidationError as e:
    print("✓ Caught error:", e.errors()[0]['msg'])
```

---

### Task 5: Implement Dataset_config.yaml Schemas
**Priority:** P0  
**Duration:** 60 min  
**Dependencies:** Task 2

Implement Pydantic models for dataset_config.yaml.

**Implementation:**
1. `TimeConfig` - dt and total_time with dt < total_time validation
2. `GenerationConfig` - num_samples, random_seed
3. `DamageConfig` - Damage settings with range validations
4. `CandidateNodesConfig` - Node selection configuration
5. `LoadGenerationConfig` - Complex load generation settings
6. `DampingConfig` - Rayleigh damping parameters
7. `TrainConfig` - Training hyperparameters
8. `InferenceConfig` - Inference settings
9. `DeepLearningConfig` - Container for train/inference
10. `DatasetConfig` - Root model

**Key Validations:**
- `dt > 0`, `total_time > dt`
- `reduction_range`: `0 < min < max < 1`
- `mode` in valid enum values
- Array length consistency (dof_candidates vs dof_weights)
- Positive values for epochs, batch_size, lr

**Verification:**
```python
from PyFEM_Dynamics.config.schemas import DatasetConfig

with open('dataset_config.yaml') as f:
    data = yaml.safe_load(f)
    
config = DatasetConfig.model_validate(data)
print(f"✓ Valid: {config.generation.num_samples} samples")
```

---

### Task 6: Create Validator Module
**Priority:** P0  
**Duration:** 45 min  
**Dependencies:** Task 4, Task 5

Create high-level validator functions and error formatters.

**Implementation in `validator.py`:**

```python
def format_validation_error(error: ValidationError, file_path: str) -> str:
    """Format Pydantic errors as bilingual messages"""
    
def validate_structure(data: dict, file_path: str = "") -> StructureConfig:
    """Validate structure.yaml data"""
    
def validate_structure_file(file_path: str) -> StructureConfig:
    """Load and validate structure.yaml file"""
    
def validate_dataset_config(data: dict, file_path: str = "") -> DatasetConfig:
    """Validate dataset_config.yaml data"""
    
def validate_dataset_config_file(file_path: str) -> DatasetConfig:
    """Load and validate dataset_config.yaml file"""
```

**Error Format:**
```
配置文件错误 [structure.yaml]
  位置: elements[5].material
  原因: 引用未定义材料 'steel2'
  可用材料: ['steel']

Config Error [structure.yaml]
  Location: elements[5].material
  Issue: References undefined material 'steel2'
  Available: ['steel']
```

**Verification:**
```python
from PyFEM_Dynamics.config.validator import validate_structure_file

config = validate_structure_file('structure.yaml')
print("✓ Validation passed")
```

---

### Task 7: Integrate Validation into io_parser.py
**Priority:** P0  
**Duration:** 30 min  
**Dependencies:** Task 6

Modify `PyFEM_Dynamics/core/io_parser.py` to call validators.

**Changes:**
1. Import validator functions at top
2. In `load_structure_yaml()`:
   - After `yaml.safe_load()`, call `validate_structure(data, file_path)`
   - Use validated data for conversion
3. In `load_dataset_config()`:
   - After `yaml.safe_load()`, call `validate_dataset_config(data, file_path)`
   - Return `validated.model_dump()`

**Backward Compatibility:**
- No changes to return types or function signatures
- Existing valid configs continue to work
- Only adds validation errors for invalid configs

**Verification:**
```bash
python -c "
from PyFEM_Dynamics.core.io_parser import YAMLParser
data = YAMLParser.load_structure_yaml('structure.yaml')
print(f'✓ Loaded: {data.num_nodes} nodes, {data.num_elements} elements')
"
```

---

### Task 8: Update Requirements
**Priority:** P0  
**Duration:** 5 min  
**Dependencies:** None

Add pydantic to requirements.txt.

**Implementation:**
```bash
# Add to requirements.txt
pydantic>=2.0.0
```

**Verification:**
```bash
pip install -r requirements.txt
python -c "import pydantic; print(f'Pydantic {pydantic.__version__}')"
```

---

### Task 9: Write Schema Unit Tests
**Priority:** P1  
**Duration:** 90 min  
**Dependencies:** Task 4, Task 5

Create comprehensive unit tests for all schemas.

**Test Files:**
```
tests/test_config/
├── __init__.py
├── test_structure_schemas.py
└── test_dataset_schemas.py
```

**Test Coverage:**
- Valid configs pass validation
- Invalid field types raise ValidationError
- Out-of-range values raise ValidationError
- Missing required fields raise ValidationError
- Extra fields raise ValidationError
- Cross-field validators catch inconsistencies
- Error messages are clear and helpful

**Example Tests:**
```python
def test_valid_structure():
    """Test valid structure.yaml passes"""
    
def test_duplicate_node_ids():
    """Test duplicate node IDs are caught"""
    
def test_undefined_material_reference():
    """Test undefined material in element is caught"""
    
def test_invalid_constraint():
    """Test invalid boundary constraint is caught"""
    
def test_negative_elastic_modulus():
    """Test negative E value is caught"""
    
def test_time_inconsistency():
    """Test total_time <= dt is caught"""
```

**Verification:**
```bash
pytest tests/test_config/ -v
```

---

### Task 10: Write Integration Tests
**Priority:** P1  
**Duration:** 45 min  
**Dependencies:** Task 7, Task 9

Create integration tests for io_parser validation.

**Test Files:**
```
tests/test_config/
└── test_io_parser_integration.py
```

**Test Coverage:**
- `load_structure_yaml()` validates before returning
- `load_dataset_config()` validates before returning
- Invalid YAML files produce clear errors
- Valid YAML files load correctly
- Error messages include file paths

**Verification:**
```bash
pytest tests/test_config/test_io_parser_integration.py -v
```

---

### Task 11: Create Test Fixtures
**Priority:** P1  
**Duration:** 30 min  
**Dependencies:** Task 9

Create YAML fixtures for testing edge cases.

**Fixtures in `tests/fixtures/config/`:**
- `valid_structure.yaml` - Copy of main structure.yaml
- `duplicate_nodes.yaml` - Duplicate node IDs
- `undefined_material.yaml` - Element references missing material
- `invalid_constraint.yaml` - Invalid boundary constraint
- `missing_metadata.yaml` - Missing required field
- `extra_field.yaml` - Unknown field present
- `negative_values.yaml` - Negative physical values
- `valid_dataset_config.yaml` - Valid dataset config
- `inconsistent_time.yaml` - total_time < dt

**Verification:**
All fixtures exist and are loadable.

---

### Task 12: Test Against Existing Configs
**Priority:** P1  
**Duration:** 20 min  
**Dependencies:** Task 7

Validate all existing YAML files in the project.

**Test:**
```bash
python -c "
from PyFEM_Dynamics.core.io_parser import YAMLParser
from PyFEM_Dynamics.config.validator import validate_dataset_config_file

# Test structure.yaml
data = YAMLParser.load_structure_yaml('structure.yaml')
print(f'✓ structure.yaml: {data.num_nodes} nodes')

# Test dataset_config.yaml
config = validate_dataset_config_file('dataset_config.yaml')
print(f'✓ dataset_config.yaml: {config.generation.num_samples} samples')
"
```

**Acceptance:**
- All existing configs pass validation
- Any failures are documented as config bugs (not validator bugs)

---

### Task 13: Run Full Test Suite
**Priority:** P1  
**Duration:** 15 min  
**Dependencies:** Task 9, Task 10

Run all tests to ensure no regressions.

**Commands:**
```bash
pytest tests/ -v --tb=short
pytest tests/ --cov=PyFEM_Dynamics.config --cov-report=term-missing
```

**Acceptance:**
- All existing tests pass
- New validation tests pass
- Coverage for config module > 80%

---

### Task 14: Document Usage
**Priority:** P2  
**Duration:** 20 min  
**Dependencies:** Task 7

Add documentation to AGENTS.md or README about validation.

**Documentation:**
- Validation is automatic when loading configs
- Error messages are bilingual (Chinese + English)
- Common validation errors and fixes
- How to bypass validation in emergencies (if needed)

---

## Task Dependencies Graph

```
Task 1 (Package Structure)
    ↓
Task 2 (Errors)
    ↓
Task 3 (Structure Base) ─────┐
    ↓                        │
Task 4 (Structure Validators)│
    ↓                        │
Task 6 (Validator Module)    │
    ↓                        │
Task 7 (io_parser Integration)│
    ↓                        │
Task 10 (Integration Tests)  │
                             │
Task 5 (Dataset Schemas) ────┤
    ↓                        │
Task 6 (Validator Module) ───┘

Task 8 (Requirements) - Parallel
Task 9 (Unit Tests) - Depends on 4, 5
Task 11 (Fixtures) - Depends on 9
Task 12 (Test Existing) - Depends on 7
Task 13 (Full Suite) - Depends on 9, 10
Task 14 (Documentation) - Depends on 7
```

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Overly strict validation breaks existing configs | High | Test all existing YAML files; use `extra='ignore'` mode if needed temporarily |
| Complex validators have bugs | Medium | Comprehensive unit tests; simple validators first |
| Performance regression | Low | Pydantic v2 is fast; validate once at load time |
| Error messages are unclear | Medium | Iterate on message format; test with example errors |
| Circular imports with io_parser | Low | Keep schemas in separate package; use forward refs |

---

## Verification Checklist

- [ ] Task 1: Package structure created and importable
- [ ] Task 2: Error classes raise bilingual messages
- [ ] Task 3: StructureConfig validates base fields
- [ ] Task 4: Cross-field validators catch inconsistencies
- [ ] Task 5: DatasetConfig validates all fields
- [ ] Task 6: Validator module provides clean API
- [ ] Task 7: io_parser.py integrates validation seamlessly
- [ ] Task 8: pydantic in requirements.txt
- [ ] Task 9: Unit tests cover all validators
- [ ] Task 10: Integration tests verify io_parser
- [ ] Task 11: Test fixtures for edge cases
- [ ] Task 12: All existing configs pass validation
- [ ] Task 13: Full test suite passes
- [ ] Task 14: Documentation updated

---

## Notes

- Use `pydantic>=2.0.0` (v2 has major performance improvements)
- Keep validators simple and focused
- Error messages should help users fix configs quickly
- Consider adding `--skip-validation` flag in future for emergency bypass
- Schema versioning can be added later if configs evolve

---

**Plan Created:** 2026-03-17  
**Ready for Execution**
