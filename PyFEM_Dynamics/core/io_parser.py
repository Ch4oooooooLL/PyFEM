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
class StructureData:
    node_coords: np.ndarray
    element_conn: np.ndarray
    E_base: np.ndarray
    rho: np.ndarray
    nu_base: np.ndarray
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
    def build_structure_objects(structure_path: str) -> Tuple[List[Node], List[TrussElement2D], List[Tuple[int, int, float]]]:
        """
        从YAML文件构建结构模型对象（Node, Element等）。
        返回: (nodes, elements, boundary_conditions)
        """
        with open(structure_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 1. 解析材料库
        materials_data = data.get('materials', [])
        materials_dict: Dict[str, Material] = {}
        for mat_info in materials_data:
            name = mat_info['name']
            E = mat_info['E']
            rho = mat_info.get('rho', 0.0)
            nu = mat_info.get('nu', 0.3)
            materials_dict[name] = Material(E=E, rho=rho, nu=nu)

        nodes_data = data['nodes']
        elements_data = data['elements']
        boundary_data = data.get('boundary', [])
        meta = data['metadata']
        dofs_per_node = meta.get('dofs_per_node', 2)
        
        nodes_dict: Dict[int, Node] = {}
        for node_info in nodes_data:
            nid = node_info['id']
            x, y = node_info['coords']
            nodes_dict[nid] = Node(nid, x, y)
        
        nodes = list(nodes_dict.values())
        nodes.sort(key=lambda n: n.node_id)
        
        dof_counter = 0
        for node in nodes:
            node.dofs = list(range(dof_counter, dof_counter + dofs_per_node))
            dof_counter += dofs_per_node
        
        elements: List[TrussElement2D] = []
        for elem_info in elements_data:
            eid = elem_info['id']
            n1_id, n2_id = elem_info['nodes']
            A = elem_info['A']
            I = elem_info.get('I', 0.0)
            
            # 引用材料
            mat_name = elem_info['material']
            if mat_name not in materials_dict:
                raise ValueError(f"Element {eid} references undefined material: {mat_name}")
            
            base_mat = materials_dict[mat_name]
            # 深度拷贝材料对象，方便后续可能的损伤修改
            mat = Material(E=base_mat.E, rho=base_mat.rho, nu=base_mat.nu)
            
            n1 = nodes_dict[n1_id]
            n2 = nodes_dict[n2_id]
            sec = Section(A=A, I=I)
            
            elem = TrussElement2D(eid, n1, n2, mat, sec)
            elements.append(elem)
        
        bcs: List[Tuple[int, int, float]] = []
        for bc_info in boundary_data:
            node_id = bc_info['node_id']
            constraints = bc_info['constraints']
            for constraint in constraints:
                if constraint == 'ux':
                    local_dof = 0
                elif constraint == 'uy':
                    local_dof = 1
                elif constraint == 'rz':
                    local_dof = 2
                else:
                    continue
                bcs.append((node_id, local_dof, 0.0))
        
        return nodes, elements, bcs

    @staticmethod
    def load_structure_yaml(file_path: str) -> StructureData:
        """
        从YAML文件加载结构定义。
        返回StructureData对象，包含结构的数值矩阵表示。
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        meta = data['metadata']
        materials_data = data.get('materials', [])
        nodes_data = data['nodes']
        elements_data = data['elements']
        boundary_data = data.get('boundary', [])
        
        num_nodes = len(nodes_data)
        num_elements = len(elements_data)
        dofs_per_node = meta.get('dofs_per_node', 2)
        num_dofs = num_nodes * dofs_per_node

        # 处理材料查找表
        mats_lookup = {m['name']: m for m in materials_data}
        
        node_coords = np.zeros((num_nodes, 2), dtype=float)
        element_conn = np.zeros((num_elements, 2), dtype=int)
        E_base = np.zeros(num_elements, dtype=float)
        rho = np.zeros(num_elements, dtype=float)
        nu_base = np.zeros(num_elements, dtype=float)
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
            A[eid] = elem['A']
            I[eid] = elem.get('I', 0.0)
            
            # 材料映射
            mat_name = elem['material']
            mat_info = mats_lookup[mat_name]
            E_base[eid] = mat_info['E']
            rho[eid] = mat_info.get('rho', 0.0)
            nu_base[eid] = mat_info.get('nu', 0.3)
        
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
            nu_base=nu_base,
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
