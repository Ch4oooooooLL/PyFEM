# Concerns

## Overview
Technical debt, limitations, and areas requiring attention in the codebase.

---

## Known Issues

### 1. No Formal Testing Framework
**Severity: High**
- No unit tests, integration tests, or CI/CD
- Verification relies on manual smoke tests
- Risk of regressions when modifying core FEM logic

**Mitigation:**
- Run smoke tests before commits
- Validate against notebooks

### 2. Limited Element Types
**Severity: Medium**
- Only `TrussElement2D` implemented
- No beam elements (despite `I` parameter in config)
- No 3D elements

**Impact:** Restricted to 2D truss structures only.

### 3. PINN Model Underperformance
**Severity: Medium**
- MAE ~0.47 vs GT MAE ~0.085
- Poor convergence (0.28% improvement over 100 epochs)
- Physics constraints conflict with data fitting

**Root Cause:** Loss weighting not adaptive; physics and data terms differ by orders of magnitude.

### 4. Memory Efficiency
**Severity: Low-Medium**
- Dense matrices in some operations could be sparse
- Full time history stored in memory during integration
- Dataset loaded entirely into RAM

**Impact:** Limits problem size for large structures or long time histories.

### 5. Configuration Duplication
**Severity: Low**
- Similar parameters in `dataset_config.yaml` and `condition_case.yaml`
- Path resolution scattered across modules
- No centralized config management

---

## Security Concerns

### No Input Validation
- YAML configs parsed without schema validation
- File paths resolved without sanitization
- No protection against path traversal

**Risk:** Low (internal tool, not exposed to external users)

### No Secrets Management
- No API keys or credentials in codebase
- Model checkpoints stored locally

**Status:** ✓ Clean

---

## Performance Concerns

### Data Generation Bottleneck
- 20,000 samples takes ~30 minutes
- Single-threaded FEM solves
- No batch processing

### Training Efficiency
- No mixed precision (AMP disabled by default)
- `num_workers=0` for DataLoader
- GPU underutilized for small batch sizes

### FEM Solver
- Sparse matrix operations could be optimized
- LU factorization recomputed each step (not needed for linear system)
- No matrix preconditioning

---

## Code Quality Issues

### 1. Limited Documentation
- No inline comments (by design per AGENTS.md)
- Chinese comments in YAML may not be accessible to all users
- Missing module-level docstrings

### 2. Error Handling Gaps
```python
# Some functions lack validation
model_type = config.get('model', 'gt')  # No validation of value
```

### 3. Hardcoded Values
```python
static_loads = [(3, 10000.0, -20000.0)]  # In main.py
scale_factor = 500.0                     # Deformation scaling
```

### 4. Cross-Module Path Hacks
```python
import sys
sys.path.append(os.path.join(ROOT_DIR, "PyFEM_Dynamics"))
```

---

## Maintainability Risks

### Tight Coupling
- `Condition_prediction` tightly coupled to both FEM and DL modules
- Path dependencies between modules
- Shared configuration structure assumptions

### Magic Numbers
- `0.005` (section area) scattered in configs
- Damping coefficients in multiple files
- Damage threshold `0.95` hardcoded

### Notebook Proliferation
- 8 notebooks with overlapping functionality
- Code duplication between notebooks and scripts
- No notebook testing or validation

---

## Recommended Improvements

### Short Term
1. Add schema validation for YAML configs
2. Extract hardcoded values to constants
3. Add type hints to remaining functions
4. Consolidate duplicate config parameters

### Medium Term
1. Implement proper test suite (pytest)
2. Add beam element support
3. Optimize data generation with multiprocessing
4. Improve PINN loss weighting strategy

### Long Term
1. Package as installable Python module
2. Add 3D element support
3. Implement proper logging (not just print)
4. Add CI/CD pipeline

---

## Technical Debt Summary

| Category | Count | Priority |
|----------|-------|----------|
| No tests | 1 | High |
| Underperforming model | 1 | Medium |
| Hardcoded values | ~10 | Low |
| Missing validation | ~5 | Medium |
| Performance issues | 3 | Low |

---

## File-Specific Concerns

| File | Issue |
|------|-------|
| `PyFEM_Dynamics/main.py` | Hardcoded loads, only for testing |
| `Deep_learning/models/pinn_model.py` | Poor convergence |
| `Condition_prediction/pipelines/condition_pipeline.py` | Complex, tightly coupled |
| `notebooks/*.ipynb` | Untested, may drift from code |
