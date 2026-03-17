# Directory Structure

## Overview
```
D:\CODE\FEM\                # Project root (~3,100 lines of Python)
├── PyFEM_Dynamics/         # FEM solver implementation
├── Deep_learning/          # Neural network models
├── Condition_prediction/   # Integration pipeline
├── notebooks/              # Jupyter analysis notebooks
├── docs/                   # Documentation and images
├── .planning/              # Codebase mapping (this directory)
├── requirements.txt        # Main dependencies
└── *.yaml                  # Configuration files
```

---

## Detailed Structure

### PyFEM_Dynamics/ — FEM Solver (~1,200 lines)
```
PyFEM_Dynamics/
├── core/                   # Domain models
│   ├── __init__.py
│   ├── node.py            # Node class (2 DOF: ux, uy)
│   ├── element.py         # Element2D ABC, TrussElement2D
│   ├── material.py        # Material properties
│   ├── section.py         # Cross-section properties
│   └── io_parser.py       # YAML structure parser
├── solver/                # Numerical methods
│   ├── __init__.py
│   ├── assembler.py       # Global K/M assembly
│   ├── boundary.py        # BC application
│   ├── integrator.py      # Newmark-β integration
│   └── stress_recovery.py # Post-processing
├── pipeline/
│   └── data_gen.py        # Dataset generation (~400 lines)
├── postprocess/
│   ├── plotter.py         # Visualization
│   └── generate_vm_cloud.py # Animation generation
└── main.py                # Static test entry point
```

### Deep_learning/ — ML Models (~800 lines)
```
Deep_learning/
├── models/
│   ├── __init__.py
│   ├── gt_model.py        # Graph Transformer (~160 lines)
│   └── pinn_model.py      # Physics-Informed NN (~150 lines)
├── data/
│   ├── __init__.py
│   └── dataset.py         # FEMDataset PyTorch dataset
├── utils/
│   ├── __init__.py
│   ├── metrics.py         # Evaluation metrics
│   └── visualization.py   # Training plots
├── checkpoints/           # Model weights (gitignored)
│   ├── gt_best.pth
│   └── pinn_best.pth
├── requirements.txt       # DL-specific deps
└── train.py               # Training script (~250 lines)
```

### Condition_prediction/ — Integration (~600 lines)
```
Condition_prediction/
├── pipelines/
│   └── condition_pipeline.py  # Main workflow
├── inference/
│   ├── __init__.py
│   └── model_inference.py     # Model inference wrapper
├── postprocess/
│   ├── __init__.py
│   └── comparison.py          # Result comparison
├── scripts/
│   └── run_condition_prediction_cli.py  # CLI entry
├── data/
│   ├── __init__.py
│   └── load_builder.py        # Load construction
├── models/
│   └── __init__.py
├── utils/
│   └── __init__.py
└── __init__.py
```

### notebooks/ — Analysis (8 notebooks)
```
notebooks/
├── 01_structure_explorer.ipynb
├── 02_dataset_overview.ipynb
├── 03_fem_simulation.ipynb
├── 04_model_training_history.ipynb
├── 05_damage_prediction.ipynb
├── 06_model_comparison.ipynb
├── 07_stress_analysis.ipynb
├── 08_interactive_dashboard.ipynb
└── utils/                 # Notebook utilities
    ├── __init__.py
    ├── data_loader.py
    ├── config.py
    └── visualizer.py
```

---

## Configuration Files

| File | Lines | Purpose |
|------|-------|---------|
| `structure.yaml` | 94 | Structure geometry, materials, boundary conditions |
| `dataset_config.yaml` | 72 | Dataset generation parameters |
| `condition_case.yaml` | ~60 | Inference case configuration |
| `load_template.yaml` | ~40 | Load pattern templates |

---

## Generated Artifacts (Gitignored)

```
dataset/
└── train.npz              # Generated training data (~20,000 samples)

postprocess_results/       # Analysis outputs
├── static_deformation.png
└── *.gif                  # Animation files

outputs/                   # General outputs
```

---

## Naming Conventions

### Files
- **Modules**: `snake_case.py` (e.g., `gt_model.py`)
- **Scripts**: `snake_case.py` with verb prefix (e.g., `run_condition_prediction_cli.py`)
- **Configs**: `snake_case.yaml`
- **Notebooks**: `NN_description.ipynb`

### Classes
- **PascalCase**: `Node`, `TrussElement2D`, `GTDamagePredictor`
- **Abstract Base**: Suffix with implied type (e.g., `Element2D`)

### Functions
- **snake_case**: `get_local_stiffness`, `train_one_epoch`
- **Private prefix**: `_resolve_path`, `_set_global_determinism`

### Variables
- **snake_case**: `element_conn`, `num_nodes`, `hidden_dim`
- **Constants**: `SCRIPT_DIR`, `ROOT_DIR`

---

## Key Locations

| Purpose | Location |
|---------|----------|
| FEM entry point | `PyFEM_Dynamics/main.py` |
| Training entry | `Deep_learning/train.py` |
| Inference entry | `Condition_prediction/scripts/run_condition_prediction_cli.py` |
| Structure config | `./structure.yaml` |
| Dataset config | `./dataset_config.yaml` |
| Inference config | `./condition_case.yaml` |
| Training data | `./dataset/train.npz` |
| GT checkpoint | `Deep_learning/checkpoints/gt_best.pth` |
| PINN checkpoint | `Deep_learning/checkpoints/pinn_best.pth` |
