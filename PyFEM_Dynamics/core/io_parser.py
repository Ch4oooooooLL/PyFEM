import csv
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from core.element import BeamElement2D, Element2D, TrussElement2D
from core.material import Material
from core.node import Node
from core.section import Section


@dataclass
class DynamicLoadRange:
    node_id: int
    fx_min: float
    fx_max: float
    fy_min: float
    fy_max: float
    t_start_min: float
    t_start_max: float
    t_end_min: float
    t_end_max: float


class IOParser:
    """
    负责解析材料、结构、静载和动载范围配置。
    """

    @staticmethod
    def load_materials(file_path: str) -> Dict[int, Material]:
        """
        读取材料属性定义的 CSV 文件。
        格式: id, E, rho
        """
        materials: Dict[int, Material] = {}
        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if not row or row[0].strip().startswith("#"):
                    continue
                if len(row) < 3:
                    raise ValueError(f"Invalid material row in {file_path}: {row}")
                mat_id = int(row[0].strip())
                e_mod = float(row[1].strip())
                rho = float(row[2].strip())
                materials[mat_id] = Material(E=e_mod, rho=rho)
        return materials

    @staticmethod
    def load_structure(
        file_path: str, materials: Dict[int, Material]
    ) -> Tuple[List[Node], List[Element2D], List[Tuple[int, int, float]], Dict[str, str]]:
        """
        读取结构定义文件。
        支持:
        - NODE, id, x, y
        - ELEM, id, type, node1_id, node2_id, mat_id, section_A, section_I
        - BC, node_id, local_dof, value
        - CONFIG, key, value
        """
        nodes_dict: Dict[int, Node] = {}
        elements: List[Element2D] = []
        bcs: List[Tuple[int, int, float]] = []
        configs: Dict[str, str] = {}

        with open(file_path, mode="r", encoding="utf-8") as f:
            for lineno, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = [p.strip() for p in line.split(",")]
                cmd = parts[0].upper()

                if cmd == "NODE":
                    if len(parts) != 4:
                        raise ValueError(f"{file_path}:{lineno} NODE expects 4 fields")
                    node_id = int(parts[1])
                    x = float(parts[2])
                    y = float(parts[3])
                    nodes_dict[node_id] = Node(node_id, x, y)
                elif cmd == "ELEM":
                    if len(parts) != 8:
                        raise ValueError(f"{file_path}:{lineno} ELEM expects 8 fields")

                    elem_id = int(parts[1])
                    elem_type = parts[2]
                    n1 = int(parts[3])
                    n2 = int(parts[4])
                    mat_id = int(parts[5])
                    sec_a = float(parts[6])
                    sec_i = float(parts[7])

                    if n1 not in nodes_dict or n2 not in nodes_dict:
                        raise ValueError(
                            f"{file_path}:{lineno} ELEM references undefined node(s): {n1}, {n2}"
                        )
                    if mat_id not in materials:
                        raise ValueError(f"{file_path}:{lineno} unknown material id: {mat_id}")

                    # 每个单元独立持有材料对象，避免损伤修改时串扰
                    base_mat = materials[mat_id]
                    mat = Material(E=base_mat.E, rho=base_mat.rho)
                    sec = Section(A=sec_a, I=sec_i)

                    if elem_type.upper() == "TRUSS2D":
                        element = TrussElement2D(
                            elem_id, nodes_dict[n1], nodes_dict[n2], mat, sec
                        )
                    elif elem_type.upper() == "BEAM2D":
                        element = BeamElement2D(
                            elem_id, nodes_dict[n1], nodes_dict[n2], mat, sec
                        )
                    else:
                        raise ValueError(f"{file_path}:{lineno} unknown element type: {elem_type}")
                    elements.append(element)
                elif cmd == "BC":
                    if len(parts) != 4:
                        raise ValueError(f"{file_path}:{lineno} BC expects 4 fields")
                    node_id = int(parts[1])
                    local_dof = int(parts[2])
                    value = float(parts[3])
                    bcs.append((node_id, local_dof, value))
                elif cmd == "CONFIG":
                    if len(parts) != 3:
                        raise ValueError(f"{file_path}:{lineno} CONFIG expects 3 fields")
                    configs[parts[1]] = parts[2]
                elif cmd in {"LOAD", "DLOAD", "SLOAD", "DLOAD_RANGE"}:
                    raise ValueError(
                        f"{file_path}:{lineno} load directives are not allowed in structure file"
                    )
                else:
                    raise ValueError(f"{file_path}:{lineno} unknown directive: {cmd}")

        nodes = list(nodes_dict.values())
        nodes.sort(key=lambda n: n.node_id)

        has_beam = any(isinstance(e, BeamElement2D) for e in elements)
        dofs_per_node = 3 if has_beam else 2

        dof_count = 0
        for node in nodes:
            node.dofs = list(range(dof_count, dof_count + dofs_per_node))
            dof_count += dofs_per_node

        return nodes, elements, bcs, configs

    @staticmethod
    def load_static_loads(file_path: str) -> List[Tuple[int, float, float]]:
        """
        读取静力载荷文件。
        支持:
        - SLOAD, node_id, Fx, Fy
        """
        static_loads: List[Tuple[int, float, float]] = []

        with open(file_path, mode="r", encoding="utf-8") as f:
            for lineno, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = [p.strip() for p in line.split(",")]
                cmd = parts[0].upper()

                if cmd != "SLOAD":
                    raise ValueError(f"{file_path}:{lineno} only SLOAD is allowed")
                if len(parts) != 4:
                    raise ValueError(f"{file_path}:{lineno} SLOAD expects 4 fields")

                node_id = int(parts[1])
                fx = float(parts[2])
                fy = float(parts[3])
                static_loads.append((node_id, fx, fy))

        return static_loads

    @staticmethod
    def load_dynamic_load_specs(
        file_path: str,
    ) -> Tuple[List[DynamicLoadRange], Dict[str, str]]:
        """
        读取动力载荷范围配置文件。
        支持:
        - CONFIG, key, value
        - DLOAD_RANGE, node_id, fx_min, fx_max, fy_min, fy_max, t_start_min, t_start_max, t_end_min, t_end_max
        """
        specs: List[DynamicLoadRange] = []
        configs: Dict[str, str] = {}

        with open(file_path, mode="r", encoding="utf-8") as f:
            for lineno, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = [p.strip() for p in line.split(",")]
                cmd = parts[0].upper()

                if cmd == "CONFIG":
                    if len(parts) != 3:
                        raise ValueError(f"{file_path}:{lineno} CONFIG expects 3 fields")
                    configs[parts[1]] = parts[2]
                    continue

                if cmd != "DLOAD_RANGE":
                    raise ValueError(
                        f"{file_path}:{lineno} only CONFIG and DLOAD_RANGE are allowed"
                    )
                if len(parts) != 10:
                    raise ValueError(f"{file_path}:{lineno} DLOAD_RANGE expects 10 fields")

                spec = DynamicLoadRange(
                    node_id=int(parts[1]),
                    fx_min=float(parts[2]),
                    fx_max=float(parts[3]),
                    fy_min=float(parts[4]),
                    fy_max=float(parts[5]),
                    t_start_min=float(parts[6]),
                    t_start_max=float(parts[7]),
                    t_end_min=float(parts[8]),
                    t_end_max=float(parts[9]),
                )

                if spec.fx_min > spec.fx_max:
                    raise ValueError(f"{file_path}:{lineno} invalid fx range")
                if spec.fy_min > spec.fy_max:
                    raise ValueError(f"{file_path}:{lineno} invalid fy range")
                if spec.t_start_min > spec.t_start_max:
                    raise ValueError(f"{file_path}:{lineno} invalid t_start range")
                if spec.t_end_min > spec.t_end_max:
                    raise ValueError(f"{file_path}:{lineno} invalid t_end range")
                if spec.t_end_max <= spec.t_start_min:
                    raise ValueError(f"{file_path}:{lineno} no feasible window where t_end > t_start")

                specs.append(spec)

        return specs, configs


@dataclass
class StructureData:
    node_coords: np.ndarray
    element_conn: np.ndarray
    E_base: np.ndarray
    rho: np.ndarray
    A: np.ndarray
    I: np.ndarray
    bc_mask: np.ndarray
    bc_values: np.ndarray
    num_nodes: int
    num_elements: int
    num_dofs: int
    dofs_per_node: int


class YAMLParser:
    """
    YAML格式的结构定义和数据集配置解析器。
    """

    @staticmethod
    def load_structure_yaml(file_path: str) -> StructureData:
        """
        从YAML文件加载结构定义。
        返回StructureData对象，包含结构的数值矩阵表示。
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        meta = data['metadata']
        nodes_data = data['nodes']
        elements_data = data['elements']
        boundary_data = data.get('boundary', [])
        
        num_nodes = len(nodes_data)
        num_elements = len(elements_data)
        dofs_per_node = meta.get('dofs_per_node', 2)
        num_dofs = num_nodes * dofs_per_node
        
        node_coords = np.zeros((num_nodes, 2), dtype=float)
        element_conn = np.zeros((num_elements, 2), dtype=int)
        E_base = np.zeros(num_elements, dtype=float)
        rho = np.zeros(num_elements, dtype=float)
        A = np.zeros(num_elements, dtype=float)
        I = np.zeros(num_elements, dtype=float)
        
        nodes_dict = {}
        for node in nodes_data:
            nid = node['id']
            nodes_dict[nid] = node
            node_coords[nid, 0] = node['coords'][0]
            node_coords[nid, 1] = node['coords'][1]
        
        for elem in elements_data:
            eid = elem['id']
            element_conn[eid, 0] = elem['nodes'][0]
            element_conn[eid, 1] = elem['nodes'][1]
            E_base[eid] = elem['E']
            rho[eid] = elem['rho']
            A[eid] = elem['A']
            I[eid] = elem.get('I', 0.0)
        
        bc_mask = np.zeros(num_dofs, dtype=bool)
        bc_values = np.zeros(num_dofs, dtype=float)
        
        for bc in boundary_data:
            node_id = bc['node_id']
            constraints = bc['constraints']
            base_dof = node_id * dofs_per_node
            
            for constraint in constraints:
                if constraint == 'ux':
                    bc_mask[base_dof + 0] = True
                elif constraint == 'uy':
                    bc_mask[base_dof + 1] = True
                elif constraint == 'rz':
                    bc_mask[base_dof + 2] = True
        
        return StructureData(
            node_coords=node_coords,
            element_conn=element_conn,
            E_base=E_base,
            rho=rho,
            A=A,
            I=I,
            bc_mask=bc_mask,
            bc_values=bc_values,
            num_nodes=num_nodes,
            num_elements=num_elements,
            num_dofs=num_dofs,
            dofs_per_node=dofs_per_node,
        )

    @staticmethod
    def load_dataset_config(file_path: str) -> Dict[str, Any]:
        """
        从YAML文件加载数据集生成配置。
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
