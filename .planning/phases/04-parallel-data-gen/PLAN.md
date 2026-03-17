# Phase 4: Parallel Data Generation - Implementation Plan

**Created:** 2026-03-17
**Status:** Ready for implementation

---

## Overview

Parallelize FEM data generation using `multiprocessing.Pool` to achieve 3-3.5x speedup on 4-core machines. Implementation modifies `data_gen.py` while preserving backward compatibility and adding `--jobs/-j` and `--seq` CLI flags.

**Target:** 20,000 samples in <10 minutes (down from 30 minutes)

---

## Task Breakdown

### Task 1: Add Dependencies
- [ ] Add `tqdm` to `requirements.txt`

**File:** `requirements.txt`
**Dependencies:** None

### Task 2: Refactor Main Function Signature
- [ ] Add `n_jobs` and `sequential` parameters to `generate_dataset()`
- [ ] Compute default `n_jobs = max(1, cpu_count() - 1)` when `n_jobs < 1`

**File:** `PyFEM_Dynamics/pipeline/data_gen.py`
**Dependencies:** None

### Task 3: Extract Sequential Logic
- [ ] Create `_generate_sequential()` function containing current loop logic
- [ ] Move lines 375-414 into new function
- [ ] Keep preallocation and metadata generation in main function

**File:** `PyFEM_Dynamics/pipeline/data_gen.py`
**Dependencies:** Task 2

### Task 4: Implement Worker Function
- [ ] Create `_generate_sample()` as top-level function (Windows picklable)
- [ ] Accept tuple argument: `(sample_idx, base_seed, config_dict)`
- [ ] Return `(sample_idx, result_dict)` where result contains load, E, disp, stress, damage
- [ ] Use `np.random.default_rng(base_seed + sample_idx)` for per-sample seeds

**File:** `PyFEM_Dynamics/pipeline/data_gen.py`
**Dependencies:** Task 3

### Task 5: Implement Parallel Execution Path
- [ ] Create `_generate_parallel()` function using `Pool.imap_unordered`
- [ ] Wrap pool iterator with `tqdm` for progress display
- [ ] Sort results by `sample_idx` after collection
- [ ] Stack arrays and write final `.npz` file

**File:** `PyFEM_Dynamics/pipeline/data_gen.py`
**Dependencies:** Task 4

### Task 6: Add CLI Arguments
- [ ] Add `--jobs/-j` argument (default: -1 for auto)
- [ ] Add `--seq` flag to disable parallelization
- [ ] Pass arguments to `generate_dataset()`

**File:** `PyFEM_Dynamics/pipeline/data_gen.py`
**Dependencies:** Task 5

### Task 7: Update Import Block
- [ ] Add `from multiprocessing import Pool, cpu_count`
- [ ] Add `from tqdm import tqdm`

**File:** `PyFEM_Dynamics/pipeline/data_gen.py`
**Dependencies:** None

### Task 8: Write Parallel Tests
- [ ] Create `tests/test_pipeline/test_parallel_data_gen.py`
- [ ] Test determinism: parallel output matches sequential with same seed
- [ ] Test small-scale execution: samples=10, jobs=2
- [ ] Test CLI argument parsing

**File:** `tests/test_pipeline/test_parallel_data_gen.py`
**Dependencies:** Tasks 1-6

---

## Dependencies Graph

```
Task 7 (imports) ──┐
Task 1 (tqdm) ─────┼──> Task 2 (signature) ──> Task 3 (extract seq) ──> Task 4 (worker)
                  │                                              │
                  └─────────────────────────────────────────────┴──> Task 5 (parallel exec)
                                                                        │
Task 8 (tests) ────────────────────────────────────────────────────────┴──> Task 6 (CLI)
```

**Parallel execution allowed:** Tasks 1, 2, 7 can run simultaneously

---

## Implementation Details

### Task 4: Worker Function

```python
def _generate_sample(args: Tuple) -> Tuple[int, Dict[str, np.ndarray]]:
    sample_idx, base_seed, structure_file, E_base, damage_config, \
    load_generation_cfg, load_mode, num_nodes, num_elements, dofs_per_node, \
    bcs, dt, total_time, damping_alpha, damping_beta = args
    
    rng = np.random.default_rng(base_seed + sample_idx)
    
    nodes_i, elements_i, bcs_i = YAMLParser.build_structure_objects(structure_file)
    
    for elem_idx, elem in enumerate(elements_i):
        elem.material.E = E_base[elem_idx]
    
    damage_vec, E_damaged = apply_damage(elements_i, damage_config, rng)
    
    if load_mode in {'random', 'random_multi_point'}:
        random_cfg = load_generation_cfg.get('random_multi_point', {})
        sample_load_specs = _build_random_load_specs(
            random_cfg=random_cfg,
            num_nodes=num_nodes,
            dofs_per_node=dofs_per_node,
            bcs=bcs_i,
            rng=rng,
        )
    else:
        sample_load_specs = load_generation_cfg.get('loads', [])
    
    num_steps = int(total_time / dt) + 1
    load_matrix = generate_load_matrix(
        sample_load_specs, num_nodes, dofs_per_node, dt, num_steps, rng
    )
    
    disp, stress = run_fem_solver(
        nodes_i, elements_i, bcs_i, load_matrix, dt, total_time,
        damping_alpha, damping_beta
    )
    
    return (
        sample_idx,
        {
            'load': load_matrix,
            'E': E_damaged,
            'disp': disp,
            'stress': stress,
            'damage': damage_vec,
        }
    )
```

### Task 5: Parallel Execution

```python
def _generate_parallel(
    config: Dict,
    n_jobs: int,
    output_file: str,
    metadata: Dict,
) -> None:
    from multiprocessing import Pool
    
    args_list = [
        (
            i,
            random_seed,
            structure_file,
            E_base,
            damage_config,
            load_generation_cfg,
            load_mode,
            num_nodes,
            num_elements,
            dofs_per_node,
            bcs,
            dt,
            total_time,
            damping_alpha,
            damping_beta,
        )
        for i in range(num_samples)
    ]
    
    results_unordered = []
    print(f"Generating {num_samples} samples with {n_jobs} workers...")
    
    with Pool(n_jobs) as pool:
        for result in tqdm(
            pool.imap_unordered(_generate_sample, args_list),
            total=num_samples,
            desc="Samples",
            unit="sample",
        ):
            results_unordered.append(result)
    
    results = sorted(results_unordered, key=lambda x: x[0])
    
    all_loads = np.stack([r[1]['load'] for r in results])
    all_E = np.stack([r[1]['E'] for r in results])
    all_disp = np.stack([r[1]['disp'] for r in results])
    all_stress = np.stack([r[1]['stress'] for r in results])
    all_damage = np.stack([r[1]['damage'] for r in results])
    
    np.savez_compressed(
        output_file,
        load=all_loads,
        E=all_E,
        disp=all_disp,
        stress=all_stress,
        damage=all_damage,
    )
    
    _save_metadata(output_file, metadata)
```

### Task 6: CLI Integration

```python
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate FEM training dataset'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to config file (default: dataset_config.yaml)'
    )
    parser.add_argument(
        '--jobs', '-j',
        type=int,
        default=-1,
        help='Number of parallel workers (default: cpu_count - 1)'
    )
    parser.add_argument(
        '--seq',
        action='store_true',
        help='Run in sequential mode (disable parallelization)'
    )
    args = parser.parse_args()
    
    generate_dataset(
        config_path=args.config,
        n_jobs=args.jobs,
        sequential=args.seq,
    )
```

---

## Verification Checklist

### Unit Tests
- [ ] `_generate_sample()` produces valid output dict with all required keys
- [ ] Seed derivation produces reproducible results
- [ ] CLI argument parsing works correctly

### Integration Tests
- [ ] Parallel mode produces same output as sequential (determinism test)
- [ ] Exception handling propagates errors correctly

### Performance Tests
- [ ] 10 samples with jobs=2 completes without error
- [ ] 100 samples with jobs=4 shows progress bar
- [ ] 1000 samples achieves >2x speedup vs sequential

### Backward Compatibility
- [ ] `generate_dataset()` works without new arguments
- [ ] Existing scripts calling `generate_dataset()` unchanged
- [ ] Sequential mode produces identical output to original implementation

---

## Success Metrics (from ROADMAP.md)

| Metric | Target | Verification |
|--------|--------|--------------|
| 4-core speedup | 3-3.5x | Benchmark: `python data_gen.py --config test.yaml` (20,000 samples) |
| 20,000 samples time | <10 minutes | Timing measurement |
| Sequential mode | Identical output | Compare .npz files with diff |
| CLI `--jobs/-j` | Works correctly | Manual test |
| CLI `--seq` | Reverts to sequential | Manual test |

---

## Test Configuration

Create minimal test config for validation:

```yaml
structure_file: structure.yaml
output_file: dataset/test.npz

time:
  dt: 0.01
  total_time: 0.5

generation:
  num_samples: 10
  random_seed: 42

load_generation:
  mode: random
  random_multi_point:
    num_loads_per_sample_range: [1, 2]
    candidate_nodes:
      mode: all_non_support_nodes
    dof_candidates: ["fy"]
    dof_weights: [1.0]
    pattern_candidates: ["harmonic"]
    pattern_weights: [1.0]
    parameter_ranges:
      F0_range: [100.0, 500.0]
      freq_range: [0.5, 2.0]
      phase: 0.0

damage:
  enabled: true
  min_damaged_elements: 1
  max_damaged_elements: 2
  reduction_range: [0.5, 0.9]

damping:
  alpha: 0.1
  beta: 0.01
```

---

## File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `requirements.txt` | Modify | Add `tqdm>=4.65.0` |
| `PyFEM_Dynamics/pipeline/data_gen.py` | Modify | Add parallel execution logic |
| `tests/test_pipeline/test_parallel_data_gen.py` | Create | New test file |

---

## Execution Order

1. **Parallel tasks (Task 1, 2, 7)** - Run simultaneously
2. **Sequential chain (Task 3 → Task 4 → Task 5 → Task 6)** - Must run in order
3. **Test file (Task 8)** - After all implementation complete

---

## Notes

- No temporary files needed - results collected in-memory then sorted
- Fail-fast error handling - exceptions propagate to main process
- Windows compatibility ensured by top-level worker function (no lambdas)
- Progress bar uses `tqdm` with `imap_unordered` for real-time updates