# Phase 2 Plan: PINN Model Optimization

## Goal-Backward Summary
**Target**: Reduce PINN MAE from 0.47 to < 0.2
**Approach**: Implement adaptive loss weighting to balance physics constraints with data terms
**Duration**: 3-4 days

## Phase Goal
解决PINN模型训练效果差的问题（MAE 0.47，100 epochs仅提升0.28%），通过实现自适应损失权重机制，使物理约束项与数据项的量级动态平衡，提升模型收敛性能。

## Current State Analysis
- **Phase 1**: COMPLETE (74 tests passing)
- **Problem**: PINN MAE stuck at 0.47 with minimal improvement
- **Root Cause**: Static loss weights cause physics/data term magnitude mismatch

## Deliverables

### 1. Adaptive Loss Weight Module
**File**: `Deep_learning/models/pinn_loss.py`  
**Contract**:
```python
class AdaptivePINNLoss(nn.Module):
    def __init__(self, physics_weight: float = 1.0, data_weight: float = 1.0, 
                 adaptive: bool = True, temperature: float = 0.1)
    def forward(self, predictions: Tensor, targets: Tensor, 
                physics_residual: Tensor) -> Tuple[Tensor, Dict[str, float]]
    # Returns: total_loss, {"data_loss": float, "physics_loss": float, 
    #                      "data_weight": float, "physics_weight": float}
```

### 2. Optimized PINN Model
**File**: `Deep_learning/models/pinn_model_v2.py`  
**Contract**:
```python
class PINNDamagePredictorV2(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 use_physics: bool = True, physics_weight: float = 0.1)
    def forward(self, x: Tensor, E: Tensor) -> Tensor
    def physics_loss(self, x: Tensor, E: Tensor, damage_pred: Tensor) -> Tensor
    # Improvements: Better initialization, residual connections, adaptive physics weighting
```

### 3. Enhanced Training Script
**File**: `Deep_learning/train_pinn_v2.py`  
**Contract**:
```python
def train_pinn_v2(config: Dict) -> Tuple[nn.Module, Dict[str, List[float]]]
    # Training loop with adaptive loss weighting
    # Returns: trained model, history dict with loss curves

def evaluate_pinn_v2(model: nn.Module, test_loader: DataLoader, 
                     device: str) -> Dict[str, float]
    # Evaluation with detailed metrics: MAE, MSE, R2, per-damage-type error
```

### 4. Experiment Tracking
**Output**: `Deep_learning/experiments/pinn_v2_results.yaml`
**Contains**: Hyperparameter grid, convergence curves, best model checkpoint path

## Task Breakdown

### Task 2.1: Research & Design (0.5 day)
**Goal**: Design adaptive loss weighting strategy  
**Depends**: None  
**Outputs**:
- Research gradient statistics-based weighting (GradNorm, DWA, or uncertainty weighting)
- Select approach and document rationale in `research/adaptive_loss_design.md`
- Define hyperparameter search space

### Task 2.2: Implement Adaptive Loss (0.5 day)
**Goal**: Create adaptive loss module  
**Depends**: Task 2.1  
**Outputs**:
- `Deep_learning/models/pinn_loss.py` with AdaptivePINNLoss
- Unit tests in `tests/test_models/test_pinn_loss.py`
- Verify loss weight updates correctly over training

### Task 2.3: Implement Optimized Model (0.5 day)
**Goal**: Build PINNDamagePredictorV2  
**Depends**: Task 2.1  
**Outputs**:
- `Deep_learning/models/pinn_model_v2.py`
- Unit tests in `tests/test_models/test_pinn_v2.py`
- Verify model outputs correct shape and gradients flow

### Task 2.4: Implement Training Script (0.5 day)
**Goal**: Create training script with adaptive weights  
**Depends**: Task 2.2, Task 2.3  
**Outputs**:
- `Deep_learning/train_pinn_v2.py`
- Integration tests in `tests/test_pipeline/test_pinn_v2_training.py`
- Config file `Deep_learning/configs/pinn_v2.yaml`

### Task 2.5: Hyperparameter Search (1 day)
**Goal**: Find optimal adaptive loss hyperparameters  
**Depends**: Task 2.4  
**Outputs**:
- Run grid search over: initial_weights, temperature, adaptation_rate
- Log results to `Deep_learning/experiments/`
- Select best configuration (target MAE < 0.2)

### Task 2.6: Validation & Documentation (0.5 day)
**Goal**: Validate improvements and document  
**Depends**: Task 2.5  
**Outputs**:
- Comparison report: v1 (baseline) vs v2 (optimized)
- Update `docs/pinn_improvements.md`
- Archive best checkpoint to `Deep_learning/checkpoints/pinn_v2_best.pth`

## Dependency Graph

```
Task 2.1 (Research)
    ↓
Task 2.2 (Loss) ──┐
                  ├──→ Task 2.4 (Training) → Task 2.5 (Tuning) → Task 2.6 (Validation)
Task 2.3 (Model) ─┘
```

## Success Criteria Checklist

### Must Have
- [ ] AdaptivePINNLoss module implemented with unit tests
- [ ] PINNDamagePredictorV2 implemented with unit tests
- [ ] train_pinn_v2.py runs without errors
- [ ] Final MAE < 0.2 on test set

### Should Have
- [ ] Convergence within 50 epochs (vs 100 for v1)
- [ ] Loss weight history visualization
- [ ] Ablation study: adaptive vs static weights
- [ ] All tests pass (`pytest tests/ -v`)

### Nice to Have
- [ ] Uncertainty-aware weighting (aleatoric/epistemic)
- [ ] Multi-scale physics constraints
- [ ] Published results to experiment tracking (wandb/mlflow)

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Adaptive weights don't converge | High | Keep v1 as fallback; try multiple strategies (GradNorm, DWA, manual annealing) |
| Training time increases | Medium | Profile loss computation; cache physics residuals |
| Overfitting to physics | Low | Monitor validation data loss separately; early stopping on data loss |

## Verification Plan

### Pre-Implementation
- [ ] Review existing PINN implementation in `Deep_learning/models/pinn_model.py`
- [ ] Understand current loss computation flow
- [ ] Confirm data loader provides necessary physics inputs

### Post-Implementation
- [ ] Run `python Deep_learning/train_pinn_v2.py --epochs 10` for smoke test
- [ ] Run full training with best config: `python Deep_learning/train_pinn_v2.py --config configs/pinn_v2_best.yaml`
- [ ] Compare MAE: baseline (0.47) vs v2 (target < 0.2)
- [ ] All unit tests pass: `pytest tests/test_models/test_pinn*.py -v`
- [ ] Integration test: `pytest tests/test_pipeline/test_pinn_v2_training.py -v`

## Phase 2 → Phase 3 Transition

### Completion Gate
Phase 2 is complete when:
1. All Must Have criteria satisfied
2. PINN v2 MAE < 0.2 demonstrated
3. Phase 2 deliverables documented in this plan
4. `.planning/phases/phase-2/VERIFICATION.md` created with results

### Handoff to Phase 3
Phase 3 (YAML Config Validation) can start in parallel after Task 2.2 completes, as it doesn't depend on PINN training results.

## State
**Status**: PLANNING  
**Created**: 2025-03-17  
**Ready for**: Task 2.1 execution
