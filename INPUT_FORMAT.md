# PyFEM-Dynamics 输入文件格式说明

项目采用三文件输入模式，不再支持旧的混合 `static_input.txt` / `batch_input.txt` 格式。

## 1. 材料文件 (`materials.csv`)
格式:

```csv
id,E,rho
0,200000000000.0,7850.0
```

## 2. 结构文件 (`structure_input.txt`)
仅用于结构与边界条件定义，支持如下指令:

- `NODE, node_id, x, y`
- `ELEM, elem_id, type, node1_id, node2_id, mat_id, A, I`
- `BC, node_id, local_dof, value`
- `CONFIG, key, value` (可选)

约束:

- 不允许出现 `SLOAD`、`LOAD`、`DLOAD`、`DLOAD_RANGE`。
- `type` 目前支持 `Truss2D` 与 `Beam2D`。

## 3. 静载文件 (`static_loads.txt`)
用于静力分析载荷输入，仅支持:

- `SLOAD, node_id, Fx, Fy`

说明:

- 载荷以全局坐标系分量给定。
- 单位默认采用 SI（N）。

## 4. 动载文件 (`dynamic_loads.txt`)
用于批量动力分析载荷与配置，支持:

- `CONFIG, key, value`
- `DLOAD_RANGE, node_id, fx_min, fx_max, fy_min, fy_max, t_start_min, t_start_max, t_end_min, t_end_max`

默认可用配置项:

- `num_samples`
- `dt`
- `total_time`
- `out_dir`
- `random_seed`
- `cloud_sample_ids` (分号分隔，如 `0;3;5`)
- `cloud_step_stride`
- `cloud_out_dir`

`DLOAD_RANGE` 约束:

- `fx_min <= fx_max`
- `fy_min <= fy_max`
- `t_start_min <= t_start_max`
- `t_end_min <= t_end_max`
- 并且必须存在可行区间满足 `t_end > t_start`

## 5. 批量输出

运行 `PyFEM_Dynamics/pipeline/data_gen.py` 后，默认输出:

- `dataset/displacement/sample_XXXXXX.csv`  
  列: `time_step,time,node_id,ux,uy`
- `dataset/stress/sample_XXXXXX.csv`  
  列: `time_step,time,element_id,sigma_axial,sigma_vm`
- `dataset/labels/damage_Y.npy`
- `dataset/meta/run_config.json`
- `postprocess_results/vm_cloud/...` (按配置抽帧的 VM 云图)

说明:

- 当前应力恢复与 VM 云图仅支持 `Truss2D`。
- 对桁架采用工程近似: `sigma_vm = abs(sigma_axial)`。
