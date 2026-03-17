# Phase 4: 数据生成并行化

## Goal
使用multiprocessing加速数据生成，20,000样本<10分钟

## Current State
- 单线程顺序执行
- 20,000样本 ~30分钟
- CPU利用率低

## Success Criteria
- [ ] 4核加速3-3.5倍
- [ ] 保留原有接口和输出格式
- [ ] 支持 `--parallel` 和 `--n-jobs` 参数
- [ ] 进度条显示正确
- [ ] 内存使用可控（不OOM）

## Approach

### 4.1 multiprocessing.Pool
```python
def generate_sample(args):
    """单个样本生成函数"""
    sample_id, config, random_seed = args
    # ... FEM simulation ...
    return result

with Pool(n_jobs) as pool:
    results = list(tqdm(
        pool.imap(generate_sample, tasks),
        total=num_samples
    ))
```

### 4.2 内存优化
- 分批次处理（每批1000个样本）
- 使用 `np.savez_compressed` 增量写入
- 或使用 `h5py` 流式写入

## Tasks

### 4.1 Refactor for parallelization
**New file**: `PyFEM_Dynamics/pipeline/data_gen_core.py`
- 提取 `generate_single_sample()` 函数
- 确保无全局状态
- 所有参数通过函数传入

### 4.2 Create parallel version
**New file**: `PyFEM_Dynamics/pipeline/data_gen_parallel.py`
```python
def generate_dataset_parallel(
    config: dict,
    num_samples: int,
    n_jobs: int = -1,
    batch_size: int = 1000,
    output_file: str = "dataset/train.npz"
):
    """并行数据生成主函数"""
```

### 4.3 Update main data_gen.py
**Modify**: `PyFEM_Dynamics/pipeline/data_gen.py`
- 添加 `--parallel` 参数
- 添加 `--n-jobs` 参数 (default: cpu_count)
- 自动选择单线程/多线程版本

### 4.4 Worker process
**New file**: `PyFEM_Dynamics/pipeline/worker.py`
```python
def worker_init():
    """工作进程初始化"""
    # 设置随机种子
    # 预加载结构配置

def worker_generate(args) -> dict:
    """工作进程主函数"""
    # 执行FEM仿真
    # 返回结果字典
```

## Performance Target

| Jobs | Expected Time | Speedup |
|------|---------------|---------|
| 1 (baseline) | 30 min | 1x |
| 2 | 16-17 min | 1.8x |
| 4 | 9-10 min | 3.2x |
| 8 | 6-7 min | 4.5x |

## Implementation Details

### Random Seed Management
```python
def get_sample_seed(base_seed: int, sample_id: int) -> int:
    """确保每个样本有确定性的唯一种子"""
    return base_seed + sample_id * 1000
```

### Progress Tracking
```python
from tqdm import tqdm

with Pool(n_jobs) as pool:
    imap_iter = pool.imap_unordered(worker_generate, tasks, chunksize=10)
    results = list(tqdm(imap_iter, total=num_samples, desc="Generating"))
```

### Error Handling
```python
def worker_generate(args):
    try:
        # ... simulation ...
        return result
    except Exception as e:
        logger.error(f"Sample {sample_id} failed: {e}")
        return None  # 跳过失败样本

# 过滤掉None结果
results = [r for r in results if r is not None]
```

## Backward Compatibility
```python
# data_gen.py 保持原接口
def generate_dataset(*args, **kwargs):
    if kwargs.get('parallel'):
        return generate_dataset_parallel(*args, **kwargs)
    else:
        return generate_dataset_sequential(*args, **kwargs)
```

## Notes
- 先确保Phase 1测试完成（验证重构正确性）
- 在Linux/Mac测试（Windows进程fork限制）
- 监控内存使用
- **Conda环境**: FEM
- 无需额外依赖（multiprocessing是标准库）
