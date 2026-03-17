# Testing

## Overview
**No formal test suite** exists. The project relies on smoke tests and validation through:
1. Manual execution of entry scripts
2. Training convergence metrics
3. FEM result verification against theoretical expectations

---

## Smoke Tests

Run these commands to verify basic functionality:

### FEM Solver
```bash
# Static analysis test
python PyFEM_Dynamics/main.py
```
Expected: Generates `postprocess_results/static_deformation.png`

### Data Generation
```bash
# Generate a small dataset (20,000 samples takes ~30 min)
python PyFEM_Dynamics/pipeline/data_gen.py
```
Expected: Creates `dataset/train.npz`

### Model Training
```bash
# Quick 1-epoch smoke test
python Deep_learning/train.py --model gt --epochs 1

# With smaller batch for debugging
python Deep_learning/train.py --model gt --epochs 1 --batch_size 8
```

### Inference Pipeline
```bash
python Condition_prediction/scripts/run_condition_prediction_cli.py --config condition_case.yaml
```

---

## Validation Approaches

### FEM Correctness
- Compare static deformation against theoretical beam theory
- Visual inspection of deformation plots
- Stress recovery consistency checks

### Model Performance
- Training loss convergence (should decrease)
- Validation MAE < 0.1 for GT model
- F1 score tracking for damage detection

### Integration
- FEM vs DL comparison metrics
- Visual alignment in animations
- Damage indicator consistency

---

## No Pytest Framework

The project does not use:
- `pytest`
- `unittest`
- Code coverage tools
- CI/CD test automation

**Testing is manual** via script execution.

---

## Debugging Tips

### Quick Debug Training
```bash
python Deep_learning/train.py --model gt --epochs 1 --batch_size 8
```

### Check Dataset
```python
import numpy as np
data = np.load('dataset/train.npz')
print(data.files)  # ['load', 'E', 'disp', 'stress', 'damage']
print(data['damage'].shape)
```

### Verify FEM Results
Run `notebooks/03_fem_simulation.ipynb` for interactive validation.

---

## Future Testing Recommendations

To add proper testing:

1. **Unit Tests** — Test individual components:
   - `Node` coordinate calculations
   - `TrussElement2D` stiffness matrices
   - `Assembler` global matrix construction

2. **Integration Tests** — End-to-end workflows:
   - YAML → FEM simulation
   - Dataset generation → Training
   - Inference pipeline completeness

3. **Regression Tests** — Known good outputs:
   - Fixed random seed FEM results
   - Trained model predictions on test cases

4. **Smoke Test Suite** — Automated validation:
   ```bash
   python -m pytest tests/smoke/ -v
   ```
