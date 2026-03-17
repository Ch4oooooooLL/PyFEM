# Phase 4: Parallel Data Generation - Research Findings

**Researched:** 2026-03-17

---

## multiprocessing.Pool patterns

### Windows Compatibility

Windows uses`spawn` method for multiprocessing (not `fork`), which requires:
1. All worker functions must be picklable (top-level module functions)
2. No lambdas or local functions as worker targets
3. Module-level imports only

```python
import multiprocessing as mp
from multiprocessing import Pool

def _generate_sample(args: Tuple) -> Tuple[int, np.ndarray, ...]:
    sample_idx, seed, config = args
    rng = np.random.default_rng(seed)
    return (sample_idx, load_matrix, E_damaged, disp, stress, damage_vec)

def generate_dataset_parallel(..., n_jobs: int = -1):
    if n_jobs < 1:
        n_jobs = max(1, mp.cpu_count() - 1)
    
    with Pool(n_jobs) as pool:
        results = list(pool.imap_unordered(_generate_sample, args_list))
```

### Key Windows Patterns

**DO:**
```python
def _worker_init(seed: int) -> None:
    np.random.seed(seed)

def _generate_sample(args: Tuple) -> Result:
    idx, data = args
    return process(data)

with Pool(n_jobs, initializer=_worker_init, initargs=(seed,)) as pool:
    results = list(pool.imap_unordered(_generate_sample, work_items))
```

**DON'T:**
```python
with Pool(n_jobs) as pool:
    results = pool.map(lambda x: process(x), items)# Lambda not picklable on Windows
```

### Context Manager Pattern

```python
from multiprocessing import Pool

def run_parallel(work_items: List, n_jobs: int):
    with Pool(n_jobs) as pool:
        for result in pool.imap_unordered(process_item, work_items):
            yield result
```

Benefits:
- Automatic cleanup on exit
- Exception-safe resource management
- Works correctly on Windows spawn

---

## tqdm + Pool integration

### Basic Pattern with imap_unordered

```python
from tqdm import tqdm
from multiprocessing import Pool

def process_sample(args):
    idx, data = args
    return idx, process(data)

work_items = [(i, data[i]) for i in range(num_samples)]

with Pool(n_jobs) as pool:
    results = list(
        tqdm(
            pool.imap_unordered(process_sample, work_items),
            total=num_samples,
            desc="Generating samples",
            unit="sample"
        )
    )
```

### Ordered Results After Completion

```python
results_unordered = []
with Pool(n_jobs) as pool:
    for result in tqdm(
        pool.imap_unordered(_generate_sample, args_list),
        total=num_samples,
        desc="Generating"
    ):
        results_unordered.append(result)

results_ordered = sorted(results_unordered, key=lambda x: x[0])
```

### Progress Bar Options (per CONTEXT.md discretion)

```python
from tqdm import tqdm

pbar = tqdm(
    total=num_samples,
    desc="Generating samples",
    unit="sample",
    ncols=100,progress characters
    mininterval=0.5,# Update frequency
)
```

---

## Temporary file management

### Pattern 1: Dedicated Output Directory

```python
import os
import tempfile

def generate_dataset_parallel(..., output_file: str):
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    temp_files = []
    for worker_id in range(n_jobs):
        temp_path = os.path.join(output_dir, f"train_worker_{worker_id}.npz")
        temp_files.append(temp_path)
```

### Pattern 2: tempfile.mkdtemp() for Isolation

```python
import tempfile
import shutil

def generate_dataset_parallel(...):
    temp_dir = tempfile.mkdtemp(prefix="data_gen_")
    temp_files = []
    
    try:
        for worker_id in range(n_jobs):
            temp_path = os.path.join(temp_dir, f"worker_{worker_id}.npz")
            temp_files.append(temp_path)
        
        with Pool(n_jobs) as pool:
            pool.map(process_batch, [(wid, temp_files[wid]) for wid in range(n_jobs)])
        
        _merge_temp_files(temp_files, output_file)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
```

### Pattern 3: In-Context.md Recommended (Worker Writes Own File)

```python
def _generate_sample_batch(args: Tuple) -> str:
    """Worker processes batch and writes to temp file, returns path."""
    worker_id, sample_indices, config, temp_dir, seed = args
    rng = np.random.default_rng(seed)
    
    results = []
    sample_idx_offset = sample_indices[0]
    
    for local_idx, global_idx in enumerate(sample_indices):
        sample_data = _generate_single_sample(global_idx, config, rng)
        results.append(sample_data)
    
    temp_path = os.path.join(temp_dir, f"worker_{worker_id}.npz")
    np.savez_compressed(
        temp_path,
        load=np.stack([r[0] for r in results]),
        E=np.stack([r[1] for r in results]),
        disp=np.stack([r[2] for r in results]),
        stress=np.stack([r[3] for r in results]),
        damage=np.stack([r[4] for r in results][:,
    )
    return temp_path

def _merge_temp_files(temp_files: List[str], output_file: str, metadata: Dict):
    all_loads = []
    all_E = []
    all_disp = []
    all_stress = []
    all_damage = []
    
    for temp_path in temp_files:
        data = np.load(temp_path)
        all_loads.append(data['load'])
        all_E.append(data['E'])
        all_disp.append(data['disp'])
        all_stress.append(data['stress'])
        all_damage.append(data['damage'])
    
    np.savez_compressed(
        output_file,
        load=np.concatenate(all_loads, axis=0),
        E=np.concatenate(all_E, axis=0),
        disp=np.concatenate(all_disp, axis=0),
        stress=np.concatenate(all_stress, axis=0),
        damage=np.concatenate(all_damage, axis=0),
    )
    
    for temp_path in temp_files:
        os.remove(temp_path)
```

### Cleanup on Failure

```python
def generate_dataset_parallel(...):
    temp_files = []
    try:
        with Pool(n_jobs) as pool:
            for temp_path in tqdm(pool.imap_unordered(_worker, args_list), ...):
                temp_files.append(temp_path)
        
        _merge_temp_files(temp_files, output_file, metadata)
    except Exception as e:
        for temp_path in temp_files:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        raise
```

---

## Random seed distribution

### Method 1: Simple Offset (CONTEXT.md Decision)

```python
def _derive_seed(base_seed: int, worker_id: int, local_idx: int) -> int:
    return base_seed + worker_id * 10000+ local_idx

def _generate_sample(args: Tuple):
    sample_idx, base_seed, config = args
    rng = np.random.default_rng(base_seed + sample_idx)
    ...
```

**Pros:** Simple, deterministic, reproducible
**Cons:** Potential correlations if seeds are too close

### Method 2: numpy SeedSequence (Recommended in specifics)

```python
from numpy.random import SeedSequence

def generate_dataset_parallel(..., random_seed: int):
    ss = SeedSequence(random_seed)
    child_seeds = ss.spawn(num_samples)
    
    work_items = [
        (i, child_seeds[i], config)
        for i in range(num_samples)
    ]
    
    with Pool(n_jobs) as pool:
        results = list(pool.imap_unordered(_generate_sample, work_items))
```

### Method 3: Per-Worker SeedSequence

```python
from numpy.random import SeedSequence

def _worker_init(base_ss: SeedSequence, worker_id: int):
    global worker_rng
    worker_rng = np.random.default_rng(base_ss.spawn(1)[0])

def generate_dataset_parallel(..., random_seed: int):
    ss = SeedSequence(random_seed)
    worker_seeds = ss.spawn(n_jobs)
    
    def init_worker(wid):
        global worker_rng
        worker_rng = np.random.default_rng(worker_seeds[wid])
    
    with Pool(n_jobs, initializer=init_worker) as pool:
        ...
```

### Recommended Implementation (Method 1 per CONTEXT.md)

```python
def generate_dataset(
    config_path: str = None,
    n_jobs: int = -1,
    sequential: bool = False,
) -> None:
    if n_jobs < 1:
        n_jobs = max(1, mp.cpu_count() - 1)
    
    if sequential or n_jobs == 1:
        return _generate_dataset_sequential(config_path, random_seed)
    
    results = []
    with Pool(n_jobs) as pool:
        args_list = [
            (i, random_seed + i, config)
            for i in range(num_samples)
        ]
        for result in tqdm(
            pool.imap_unordered(_generate_sample, args_list),
            total=num_samples,
            desc="Generating samples"
        ):
            results.append(result)
```

---

## Existing code patterns

### data_gen.py Sample Loop (lines 375-414)

```python
for i in range(num_samples):
    nodes_i, elements_i, bcs_i = YAMLParser.build_structure_objects(structure_file)
    
    for elem_idx, elem in enumerate(elements_i):
        elem.material.E = E_base[elem_idx]
    
    damage_vec, E_damaged = apply_damage(elements_i, damage_config, rng)
    
    if load_mode in {'random', 'random_multi_point'}:
        random_cfg = load_generation_cfg.get('random_multi_point', {})
        sample_load_specs = _build_random_load_specs(...)
    elif load_mode in {'fixed', 'manual'}:
        sample_load_specs = load_generation_cfg.get('loads', [])
    
    load_matrix = generate_load_matrix(...)
    disp, stress = run_fem_solver(...)
    
    all_loads[i] = load_matrix
    all_E[i] = E_damaged
    all_disp[i] = disp
    all_stress[i] = stress
    all_damage[i] = damage_vec
    
    if (i + 1) % 100 == 0 or (i + 1) == num_samples:
        print(f"  进度: {i + 1}/{num_samples}")
```

**Key observations:**
- Each sample is independent (embarrassingly parallel)
- `YAMLParser.build_structure_objects()` creates fresh copies
- `apply_damage()` modifies elements in-place
- All state is captured per-sample

### train.py CLI Argument Pattern (lines 378-402)

```python
parser = argparse.ArgumentParser(description='Train damage detection models')
parser.add_argument('--config', type=str, default=os.path.join(ROOT_DIR, 'dataset_config.yaml'))
parser.add_argument('--model', type=str, default=None, choices=['gt', 'pinn', 'both'])
parser.add_argument('--epochs', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=None)
# ... more arguments
parser.add_argument('--num_workers', type=int, default=None)
parser.add_argument('--eval_only', action='store_true')
args = parser.parse_args()
```

**Pattern for --jobs/-j and --seq:**
```python
parser.add_argument('--jobs', '-j', type=int, default=-1,
                    help='Number of parallel workers (default: cpu_count-1)')
parser.add_argument('--seq', action='store_true',
                    help='Run in sequential mode (disable parallelization)')
```

### train.py Determinism Pattern (lines 95-122)

```python
def _set_global_determinism(
    seed: int,
    deterministic: bool = True,
    deterministic_warn_only: bool = False,
    num_threads: int | None = 1,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if num_threads is not None and num_threads > 0:
        torch.set_num_threads(int(num_threads))

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

### train.py Worker Seed Function (lines 137-142)

```python
def _seed_worker(worker_id: int, base_seed: int) -> None:
    """Top-level worker init function so it is picklable on spawn-based platforms."""
    worker_seed = int(base_seed) + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
```

**Note:** This pattern is already in the codebase and follows Windows-compatible design.

### data_gen.py RNG Usage (line 343)

```python
rng = np.random.default_rng(random_seed)
```

**Note:** Modern NumPy recommendation - use `default_rng()` instead of legacy `np.random.seed()`.

---

## Recommendations

Based on CONTEXT.md decisions and research:

### 1. Worker Function Design

```python
def _generate_sample(args: Tuple) -> Tuple[int, dict]:
    sample_idx: int
    base_seed: int
    structure_file: str
    E_base: np.ndarray
    damage_config: dict
    load_config: dict
    dt: float
    total_time: float
    damping_alpha: float
    damping_beta: float
    num_nodes: int
    num_elements: int
    dofs_per_node: int
    bcs: List[Tuple[int, int, float]]
    load_mode: str
    
    (
        sample_idx, base_seed, structure_file, E_base,
        damage_config, load_config, dt, total_time,
        damping_alpha, damping_beta, num_nodes, num_elements,
        dofs_per_node, bcs, load_mode
    ) = args
    
    rng = np.random.default_rng(base_seed + sample_idx)
    
    nodes_i, elements_i, bcs_i = YAMLParser.build_structure_objects(structure_file)
    
    for elem_idx, elem in enumerate(elements_i):
        elem.material.E = E_base[elem_idx]
    
    damage_vec, E_damaged = apply_damage(elements_i, damage_config, rng)
    
    if load_mode in {'random', 'random_multi_point'}:
        sample_load_specs = _build_random_load_specs(
            load_config.get('random_multi_point', {}),
            num_nodes, dofs_per_node, bcs_i, rng
        )
    else:
        sample_load_specs = load_config.get('loads', [])
    
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

### 2. Main Parallel Execution

```python
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def generate_dataset(
    config_path: str = None,
    n_jobs: int = -1,
    sequential: bool = False,
) -> None:
    if config_path is None:
        config_path = os.path.join(ROOT_DIR, 'dataset_config.yaml')
    
    # ... config loading (same as before) ...
    
    if n_jobs < 1:
        n_jobs = max(1, cpu_count() - 1)
    
    if sequential or n_jobs == 1:
        return _generate_sequential(
            config, nodes, elements, bcs, E_base, rng,
            num_samples, num_steps, num_dofs, num_elements,
            output_file, metadata
        )
    args_list = [
        (
            i, random_seed, structure_file, E_base,
            damage_config, load_generation_cfg, dt, total_time,
            alpha, beta, num_nodes, num_elements, dofs_per_node, bcs, load_mode
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
            unit="sample"
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
    
    # ... metadata saving (same as before) ...
```

### 3. CLI Integration

```python
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate FEM training dataset')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to config file (default: dataset_config.yaml)')
    parser.add_argument('--jobs', '-j', type=int, default=-1,
                        help='Number of parallel workers (default: cpu_count-1)')
    parser.add_argument('--seq', action='store_true',
                        help='Run in sequential mode (disable parallelization)')
    args = parser.parse_args()
    
    generate_dataset(
        config_path=args.config,
        n_jobs=args.jobs,
        sequential=args.seq,
    )
```

### 4. Memory Consideration

For large datasets, consider batch processing instead of collecting all results in memory:

```python
temp_dir = tempfile.mkdtemp(prefix="data_gen_")
temp_files = []

batch_size = num_samples // n_jobs
for batch_idx in range(n_jobs):
    start_idx = batch_idx * batch_size
    end_idx = start_idx + batch_size if batch_idx < n_jobs - 1 else num_samples
    
    batch_args = args_list[start_idx:end_idx]
    temp_path = os.path.join(temp_dir, f"batch_{batch_idx}.npz")
    temp_files.append(temp_path)

# Process batches in parallel, merge at end
```

### 5. Test Strategy

```python
def test_parallel_determinism():
    config_path = "test_config.yaml"
    
    generate_dataset(config_path, n_jobs=1, sequential=True)
    data_seq = np.load("dataset/train.npz")
    
    generate_dataset(config_path, n_jobs=2, sequential=False)
    data_par = np.load("dataset/train.npz")
    
    for key in ['load', 'E', 'disp', 'stress', 'damage']:
        assert np.allclose(data_seq[key], data_par[key]), f"{key} mismatch"
```

---

## Summary

| Decision | Implementation |
|----------|---------------|
| Parallel strategy | `multiprocessing.Pool` with `imap_unordered` |
| Progress display | `tqdm` wrapper around pool iterator |
| Seed distribution | `base_seed + sample_idx` (simple offset) |
| Result collection | In-memory collection + sorted by index |
| CLI args | `--jobs/-j` for workers, `--seq` for sequential |
| Windows compatibility | Top-level worker function, no lambdas |
| Error handling | Let exceptions propagate (fail fast) |
| Cleanup | No temp files needed for in-memory approach |

**Performance target:** 4-core machine should achieve 3-3.5x speedup over sequential.