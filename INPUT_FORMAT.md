# PyFEM-Dynamics 输入文件格式说明

本项目支持通过外部配置文件的方式动态构建有限元模型，从而消除代码中的硬编码限制。分析模型所需的基础输入文件分为以下三种类型。

## 1. 材料属性文件 (`materials.csv`)
该文件使用轻量级 CSV 格式独立定义各类材料的物理参数。
**格式：**
```csv
id,E,rho
[材料编号_int], [弹性模量_float], [密度_float]
```
**示例：**
```csv
id,E,rho
0,200000000000.0,7850.0
```

## 2. 单次运行模型文件 (如 `static_input.txt`)
该文件采用纯文本（TXT）关键字解析语法，用于单独定义静态分析或单个模型的节点、拓扑单元以及边界和载荷。每行按逗号分隔，`#` 开头的行为注释。

支持的指令关键字：
- **节点定义**：`NODE, [node_id_int], [x_float], [y_float]`
- **单元定义**：`ELEM, [elem_id_int], [type_str], [n1_id], [n2_id], [mat_id], [sec_A_float], [sec_I_float]`
  - `type_str` 目前支持 `Truss2D` (二维桁架) 或是 `Beam2D` (二维梁)
  - `mat_id` 必须存在于对应的 `materials.csv` 中
- **边界条件约束**：`BC, [node_id], [local_dof_int], [value]`
  - 局部自由度 `local_dof` 取值说明：`0` 代表 X 方向位移，`1` 代表 Y 方向位移，`2` 代表转角 theta
- **常量点力载荷**：`LOAD, [node_id], [local_dof_int], [value]`

## 3. 批量生成动力学运行文件 (如 `batch_input.txt`)
包含在 `static_input.txt` 中的所有的模型结构定义指令。除此之外，它添加了用于运行动力学时程分析以及机器学习批量自动管线的特有指令：

- **运行相关配置参数**：`CONFIG, [key], [value]`
  - 示例包括 `CONFIG, num_samples, 10` (生成组数)，`CONFIG, dt, 0.01` (时间步长)，`CONFIG, total_time, 2.0` (分析总时间)，`CONFIG, out_dir, dataset` (输出路径)，`CONFIG, sensor_nodes, 2;3` (指定需要记录输出响应的监控传感器节点ID集合，利用分号分割)
- **瞬态时程序列载荷矩形脉冲**：`DLOAD, [node_id], [local_dof_int], [force_value], [start_time_sec], [end_time_sec]`
