# Architecture

## Overview
Three-module architecture combining classical FEM simulation with modern deep learning for structural damage identification.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FEM + DL Pipeline                        │
├──────────────────┬──────────────────┬──────────────────────┤
│ PyFEM_Dynamics   │  Deep_learning   │ Condition_prediction │
│   (FEM Solver)   │   (ML Models)    │   (Integration)      │
├──────────────────┼──────────────────┼──────────────────────┤
│ • Core classes   │ • GT Model       │ • Inference pipeline │
│ • Solver engine  │ • PINN Model     │ • Multi-source eval  │
│ • Data gen       │ • Training       │ • Comparison viz     │
└──────────────────┴──────────────────┴──────────────────────┘
```

---

## Module 1: PyFEM_Dynamics (FEM Solver)

### Layer Structure
```
core/           # Domain objects
├── node.py     # Node class (id, x, y, dofs)
├── element.py  # Element2D ABC, TrussElement2D
├── material.py # Material properties (E, rho, nu)
├── section.py  # Cross-section properties (A, I)
└── io_parser.py # YAML config parsing

solver/         # Numerical methods
├── assembler.py    # Global K/M matrix assembly
├── boundary.py     # BC application (zero-one method)
├── integrator.py   # Newmark-β time integration
└── stress_recovery.py # Post-processing

pipeline/
└── data_gen.py     # Dataset generation workflow

postprocess/
├── plotter.py      # Visualization utilities
└── generate_vm_cloud.py # Animation generation
```

### Key Design Patterns
- **Abstract Base Class**: `Element2D` defines interface for all element types
- **Property Decorators**: Computed attributes like `length`, `angle`
- **Sparse Matrices**: `scipy.sparse` for efficient global system storage
- **Coordinate Transformation**: Local ↔ Global via transformation matrices

---

## Module 2: Deep_learning (Neural Networks)

### Layer Structure
```
models/
├── gt_model.py     # Graph Transformer architecture
├── pinn_model.py   # Physics-Informed Neural Network
└── __init__.py

data/
├── dataset.py      # FEMDataset (PyTorch Dataset)
└── __init__.py

utils/
├── metrics.py      # MAE, RMSE, F1 calculation
├── visualization.py # Training history plots
└── __init__.py

train.py            # Main training script
```

### GT Model Architecture
```
Input: (batch, seq_len, node_features)
    ↓
Node Encoder (1D Conv) → (batch, num_nodes, hidden_dim)
    ↓
Positional Encoding
    ↓
Graph Transformer Layers (Multi-Head Self-Attention)
    ↓
Edge Feature Extraction (node pair concat)
    ↓
Damage Prediction (per element)
```

### PINN Model Architecture
- Physics constraints embedded in loss function
- Smoothness regularization
- Range constraints for damage factors

---

## Module 3: Condition_prediction (Integration)

### Layer Structure
```
pipelines/
└── condition_pipeline.py  # Main integration workflow

inference/
└── model_inference.py     # Model loading & inference

postprocess/
└── comparison.py          # Multi-source result comparison

scripts/
└── run_condition_prediction_cli.py  # CLI entry point

data/
└── load_builder.py        # Load case construction
```

### Workflow
1. Load configuration from `condition_case.yaml`
2. Build deterministic load time history
3. Run FEM simulation (healthy + damaged)
4. Calculate FEM damage indicators
5. Run DL model inference
6. Generate comparison metrics and visualizations

---

## Data Flow

### Training Phase
```
YAML Config → data_gen.py → FEM Simulation → train.npz
                                               ↓
                          train.py ← FEMDataset ← DataLoader
                                               ↓
                                       Model Training
                                               ↓
                                       Checkpoint (.pth)
```

### Inference Phase
```
condition_case.yaml
    ↓
ConditionPipeline
    ↓
┌─────────────┬──────────────┬─────────────────┐
↓             ↓              ↓                 ↓
FEM (Ref)   FEM (Damage)   GT Model        PINN Model
    ↓             ↓              ↓                 ↓
Stress      Stress         Predictions     Predictions
    ↓             ↓              ↓                 ↓
└─────────────┴──────────────┴─────────────────┘
    ↓
comparison.py → Metrics + Visualizations
```

---

## Entry Points

| Script | Purpose | Command |
|--------|---------|---------|
| `PyFEM_Dynamics/main.py` | Static analysis smoke test | `python PyFEM_Dynamics/main.py` |
| `PyFEM_Dynamics/pipeline/data_gen.py` | Generate training data | `python PyFEM_Dynamics/pipeline/data_gen.py` |
| `Deep_learning/train.py` | Train models | `python Deep_learning/train.py --model gt --epochs 100` |
| `Condition_prediction/scripts/run_condition_prediction_cli.py` | Run inference | `python Condition_prediction/scripts/run_condition_prediction_cli.py --config condition_case.yaml` |

---

## Key Abstractions

### FEM Domain
- **Node**: Spatial point with coordinates and DOF mapping
- **Element2D**: Abstract element with local/global stiffness
- **Material**: Elastic properties (E, ρ, ν)
- **Assembler**: Builds global system matrices
- **BoundaryCondition**: Applies constraints via zero-one method

### ML Domain
- **FEMDataset**: PyTorch Dataset for NPZ files
- **GTDamagePredictor**: Graph neural network model
- **PINNDamagePredictor**: Physics-constrained model

### Integration Domain
- **ConditionPipeline**: Orchestrates FEM + DL workflow
- **LoadBuilder**: Constructs load time series
- **ModelInference**: Wrapper for model loading/prediction
