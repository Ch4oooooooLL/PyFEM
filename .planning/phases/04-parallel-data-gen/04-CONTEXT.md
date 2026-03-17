# Phase 4: 数据生成并行化 - Context

**Gathered:** 2026-03-17
**Status:** Ready for planning

<domain>
## Phase Boundary

使用multiprocessing加速数据生成，将20,000样本的生成时间从30分钟降到<10分钟。交付物为修改后的`data_gen.py`，保持原有接口向后兼容，添加`--jobs/-j`参数控制并行度。成功标准：4核加速3-3.5倍。

</domain>

<decisions>
## Implementation Decisions

### 并行策略与数据分片
- **方案**: `multiprocessing.Pool` — Windows平台性能最好，API简洁稳定
- **分片方式**: 预分片，每个worker处理`num_samples // n_jobs`个样本
- **结果合并**: 临时文件 + 最终合并 — 每个worker写入独立.npz文件，主进程concat后删除临时文件
- **默认并行度**: `auto` = `cpu_count() - 1`，保留一个核心给系统
- **随机种子**: `base_seed + worker_id + local_index` — 保证各worker生成独立且可复现的样本

### 错误处理与容错
- **Worker失败策略**: 立即失败并抛出异常 — 与当前单线程行为一致
- **临时文件清理**: 失败时删除所有临时文件，避免残留
- **日志策略**: 打印到控制台 — 包含完整错误栈和失败样本索引

### 进度追踪与用户体验
- **进度显示**: 单条进度条 — 使用`Pool.imap_unordered`逐步获取结果并结合tqdm显示
- **输出格式**: 与单线程版本相同 — 保持控制台输出一致性
- **依赖**: 添加`tqdm`到requirements — 仅并行部分需要，用于进度条显示

### 接口设计与兼容性
- **CLI参数命名**: `--jobs, -j` — 简短明确，与scikit-learn等库一致
- **触发方式**: 默认启用并行，`--seq`退回串行模式 — 优化向前，不影响现有工作流
- **文件结构**: 单文件实现 — `data_gen.py`自包含并行逻辑，私有函数保持模块化
- **测试策略**: 小规模并行测试 — `samples=10, jobs=2`验证正确性，固定种子比对输出值
- **向后兼容**: 保持默认参数，默认并行 — 不破坏现有脚本和工作流

### OpenCode's Discretion
- 临时文件命名格式（如`train_worker_0.npz`）
- 具体的tqdm进度条样式和更新频率
- 错误信息的具体措辞
- 内存限制的具体阈值（如`imax_bytes`参数）

</decisions>

<specifics>
## Specific Ideas

- 参考当前`data_gen.py:375-414`的循环结构，改为`Pool.map`或`Pool.imap_unordered`
- 临时文件目录使用`tempfile.mkdtemp()`或配置文件中指定的输出目录
- 样本索引分片时考虑不整除情况，最后一组可能多1-2个样本
- 使用`numpy.SeedSequence`可更科学地分布随机种子，但当前方案已足够

</specifics>

<canonical_refs>
## Canonical References

### Phase Specification
- `.planning/ROADMAP.md` §Phase 4 — 阶段目标、交付物、成功标准
- `.planning/STATE.md` — 当前性能瓶颈：20,000样本30分钟，单线程

### Codebase Context
- `.planning/codebase/STRUCTURE.md` — 代码库结构，模块说明
- `.planning/codebase/CONVENTIONS.md` — 编码规范，命名约定

### Prior Phase Context
- `.planning/phases/01-2-3/01-CONTEXT.md` — 测试框架决策（pytest使用）
- `.planning/phases/03-yaml-validation/03-CONTEXT.md` — 配置验证决策

### Existing Code
- `PyFEM_Dynamics/pipeline/data_gen.py` — 现有单线程实现
- `dataset_config.yaml` — 数据生成配置示例
- `Deep_learning/train.py` — CLI参数模式参考（`--model`, `--epochs`等）

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **generate_load_matrix()** (`data_gen.py:163-237`): 载荷生成函数，可直接在worker中调用
- **run_fem_solver()** (`data_gen.py:268-318`): FEM求解器，无状态依赖
- **apply_damage()** (`data_gen.py:240-265`): 损伤施加函数，需传入独立的elements副本
- **YAMLParser.build_structure_objects()** (`io_parser.py`): 每个样本需独立调用

### Established Patterns
- **CLI参数解析**: `train.py:378-402`的argparse模式，添加新参数应遵循相同风格
- **类型注解**: 全代码库使用现代Python类型注解，新代码需保持一致
- **配置加载**: `yaml.safe_load()` + 路径解析模式

### Integration Points
- **data_gen.py修改点**:
  - 主函数`generate_dataset()`添加`parallel`和`n_jobs`参数
  - 新增`_generate_sample()`函数作为worker入口
  - 新增`_merge_temp_files()`函数合并结果
  - 新增`_worker_init()`初始化worker随机状态
- **requirements.txt**: 添加`tqdm`依赖
- **pytest测试**: 新增`tests/test_pipeline/test_parallel_data_gen.py`

### Parallelization Candidates
```python
# 当前循环结构 (data_gen.py:375-414)
for i in range(num_samples):
    nodes_i, elements_i, bcs_i = YAMLParser.build_structure_objects(structure_file)
    for elem_idx, elem in enumerate(elements_i):
        elem.material.E = E_base[elem_idx]
    damage_vec, E_damaged = apply_damage(elements_i, damage_config, rng)
    load_matrix = generate_load_matrix(...)
    disp, stress = run_fem_solver(...)
    all_loads[i] = load_matrix
    # ...

# 并行化后结构
def _generate_sample(args):
    sample_idx, structure_file, E_base, ... = args
    # 每个样本独立生成
    return (sample_idx, load_matrix, E_damaged, disp, stress, damage_vec)

with Pool(n_jobs) as pool:
    results = list(tqdm(pool.imap_unordered(_generate_sample, args_list), total=num_samples))
```

</code_context>

<deferred>
## Deferred Ideas

- 分布式计算支持（多机器）— 远期阶段
- GPU加速FEM求解器 — 远期阶段
- 断点续传功能 — 未来阶段
- 实时监控面板 — 未来阶段

</deferred>

---

*Phase: 04-parallel-data-gen*
*Context gathered: 2026-03-17*