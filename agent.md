# Agent Guide (PyFEM-Dynamics)

This repository is a Python finite-element (FEM) solver + a small deep-learning sidecar for damage identification. This file is written to help an autonomous coding agent navigate the codebase safely, run the main flows, and make changes without breaking numerical assumptions.

Important: do not commit this file. Keep it untracked.

## 0) Repo Snapshot

Top-level (high signal):

- `PyFEM_Dynamics/`: FEM solver implementation
- `Deep_learning/`: LSTM/PINN training code consuming `dataset/train.npz`
- `README.md`: theory overview + how to run + data format
- `INPUT_FORMAT.md`: input formats (legacy txt/csv + YAML mode)
- `spec.md`: requirements/specification (module structure, solver choices)
- `docs/fem_dev_process_blog.md`: development notes walking formula -> code
- `dataset_config.yaml`, `structure.yaml`, `load_template.yaml`: YAML-mode configs
- `materials.csv`, `structure_input.txt`, `static_loads.txt`: legacy-mode example inputs

Git hygiene note:

- `.gitignore` ignores `dataset/` and `postprocess_results/` (generated artifacts).
- The working tree may already be dirty (lots of generated files / images). Do not delete or revert unrelated user changes.

## 1) Tech Stack

- Language: Python (spec targets 3.10+)
- Numerics: `numpy`, `scipy.sparse`, `scipy.sparse.linalg`
- I/O: `csv`, `pyyaml`
- Plotting: `matplotlib`
- ML (optional): `torch`, `tqdm`

No packaging/CI framework is present at repo root (no `pyproject.toml`, no `pytest.ini`, no GitHub Actions workflows). Most workflows are "run scripts directly".

## 2) Primary Entry Points (What to Run)

### Static analysis (legacy input format)

- `PyFEM_Dynamics/main.py`
  - Loads: `materials.csv`, `structure_input.txt`, `static_loads.txt`
  - Builds global stiffness, applies BCs, solves `K U = F`, plots deformation.

Run:

```bash
python PyFEM_Dynamics/main.py
```

### Dataset generation (YAML mode, used for ML)

- `PyFEM_Dynamics/pipeline/data_gen.py`
  - Reads `dataset_config.yaml` which points to `structure.yaml`
  - Generates random loads + random damage (E reduction), runs dynamics (Newmark-beta)
  - Writes `dataset/train.npz` and `dataset/metadata.json`

Run:

```bash
python PyFEM_Dynamics/pipeline/data_gen.py
```

### Deep learning training (optional)

- `Deep_learning/train.py`
  - Trains LSTM and/or PINN models on `dataset/train.npz`

Run examples:

```bash
python Deep_learning/train.py --model lstm --epochs 100
python Deep_learning/train.py --model pinn --epochs 100
python Deep_learning/train.py --model both --epochs 100
```

Dependency install for ML sidecar:

```bash
pip install -r Deep_learning/requirements.txt
```

## 3) Architecture Map (Where Things Live)

### FEM core domain model (`PyFEM_Dynamics/core/`)

- `PyFEM_Dynamics/core/node.py`: `Node` (id, x, y, global dof indices)
- `PyFEM_Dynamics/core/material.py`: `Material` (E, rho)
- `PyFEM_Dynamics/core/section.py`: `Section` (A, I)
- `PyFEM_Dynamics/core/element.py`:
  - `Element2D` base class
  - `TrussElement2D` (2 dof/node: ux, uy)
  - `BeamElement2D` (3 dof/node: ux, uy, rz)
- `PyFEM_Dynamics/core/io_parser.py`:
  - `IOParser` (legacy txt/csv format)
  - `YAMLParser` (YAML structure/config to numeric arrays)

Key invariants:

- DOF assignment:
  - Legacy `IOParser.load_structure(...)` decides `dofs_per_node` by whether any `BeamElement2D` exists.
  - YAML path in `pipeline/data_gen.py` assigns dofs per node from `metadata.dofs_per_node`.

### FEM solver (`PyFEM_Dynamics/solver/`)

- `PyFEM_Dynamics/solver/assembler.py`: assemble global `K` and `M` (sparse)
- `PyFEM_Dynamics/solver/boundary.py`: Dirichlet BC application
  - zero-one substitution (used by `main.py`)
  - penalty method (used by `pipeline/data_gen.py`)
- `PyFEM_Dynamics/solver/integrator.py`:
  - `StaticSolver.solve(K, F)`
  - `NewmarkBetaSolver` for transient dynamics (Rayleigh damping, LU factorization)
- `PyFEM_Dynamics/solver/stress_recovery.py`: truss stress recovery from displacement history

Numerical flow (dynamic):

1) assemble `K`, `M`
2) compute `C = alpha*M + beta*K`
3) apply BCs to `K`, `M`, and force history `F_t`
4) solve for `U(t)` via Newmark-beta
5) recover stresses from `U(t)`

### Postprocess (`PyFEM_Dynamics/postprocess/`)

- `PyFEM_Dynamics/postprocess/plotter.py`: structure deformation + time histories + sample plots

### Deep learning sidecar (`Deep_learning/`)

- `Deep_learning/data/dataset.py`: `FEMDataset` loads `train.npz` (supports `mode=response|load|both`)
- `Deep_learning/models/lstm_model.py`: BiLSTM + attention predictor
- `Deep_learning/models/pinn_model.py`: PINN-like predictor + composite loss
- `Deep_learning/utils/metrics.py`: metrics + early stopping
- `Deep_learning/utils/visualization.py`: training plots/reports

## 4) Data Formats (What Files Mean)

### Legacy (static demo)

- `materials.csv`: material table: `id,E,rho`
- `structure_input.txt`: directives `NODE`, `ELEM`, `BC`, `CONFIG`
- `static_loads.txt`: directives `SLOAD, node_id, Fx, Fy`

See: `INPUT_FORMAT.md` for the full grammar.

### YAML (dataset generation)

- `structure.yaml`:
  - `metadata`: `num_nodes`, `num_elements`, `dofs_per_node`, ...
  - `nodes`: list of `{id, coords}`
  - `elements`: list of `{id, nodes, E, rho, A, I}`
  - `boundary`: list of `{node_id, constraints}` where constraints are `ux|uy|rz`
- `dataset_config.yaml`:
  - time (`dt`, `total_time`)
  - generation (`num_samples`, `random_seed`)
  - damage (E reduction range)
  - load generation specs (pulse/half_sine/etc.)
- Outputs:
  - `dataset/train.npz` with keys: `load`, `E`, `disp`, `stress`, `damage`
  - `dataset/metadata.json` with shapes and config summary

## 5) Common Agent Tasks (How to Change Things Safely)

### When editing solver math

- Prefer surgical changes; preserve matrix shapes and DOF ordering.
- Never change a sign/transpose without tracing its impact through assembly + stress recovery.
- If you touch BC logic, verify both code paths:
  - static: `BoundaryCondition.apply_zero_one_method(...)`
  - dynamic: `BoundaryCondition.apply_penalty_method(...)` + `BoundaryCondition.apply_to_M(...)`

### When adding a new load pattern

- Update:
  - `PyFEM_Dynamics/pipeline/data_gen.py` (`generate_load_matrix`)
  - `load_template.yaml` documentation table
- Keep output `load_matrix` shape `(num_steps, num_dofs)`.

### When modifying data schema

- If you change keys/shapes written by `data_gen.py`, update:
  - `Deep_learning/data/dataset.py`
  - Any training/visualization code expecting existing keys.

## 6) Verification (Because There Is No Test Suite)

There is no formal test runner in this repo. Use these pragmatic checks:

1) Static smoke test:

```bash
python PyFEM_Dynamics/main.py
```

2) Dataset generation (consider temporarily reducing sample count in `dataset_config.yaml` locally if runtime is large):

```bash
python PyFEM_Dynamics/pipeline/data_gen.py
```

3) ML training smoke test (after installing deps):

```bash
python Deep_learning/train.py --model lstm --epochs 1
```

If a change affects only formatting/docs, skip runtime checks.

## 7) Gotchas / Traps

- Generated artifacts are large/noisy:
  - `dataset/`, `postprocess_results/`, `Deep_learning/checkpoints/`, `Deep_learning/figures/`.
- Mixed workflows:
  - legacy static demo uses txt/csv; pipeline uses YAML.
- DOF mapping assumptions:
  - Many routines assume 2 dof/node for truss; beam adds a 3rd dof.
- Boundary conditions in dynamics:
  - `data_gen.py` forces constrained DOFs in `F_t` and also modifies `K`/`M`.

## 8) Minimal "Where to Look" Index

- Inputs: `INPUT_FORMAT.md`, `structure_input.txt`, `structure.yaml`, `dataset_config.yaml`
- Parsing: `PyFEM_Dynamics/core/io_parser.py`
- Elements: `PyFEM_Dynamics/core/element.py`
- Assembly: `PyFEM_Dynamics/solver/assembler.py`
- BCs: `PyFEM_Dynamics/solver/boundary.py`
- Dynamics: `PyFEM_Dynamics/solver/integrator.py`
- Stress recovery: `PyFEM_Dynamics/solver/stress_recovery.py`
- Dataset pipeline: `PyFEM_Dynamics/pipeline/data_gen.py`
- ML dataset loader: `Deep_learning/data/dataset.py`

## 9) Agent Operating Rules (Project-Specific)

- Do not commit `agent.md`.
- Avoid editing `dataset/` contents; treat it as generated.
- Prefer to keep changes within one module unless the change is schema-level.
- When you must touch multiple modules, update producer/consumer together (e.g., `data_gen.py` + `Deep_learning/data/dataset.py`).
