import csv
from typing import List, Tuple, Dict
from core.node import Node
from core.material import Material
from core.section import Section
from core.element import Element2D, TrussElement2D, BeamElement2D

class IOParser:
    """
    负责从外部文件（CSV、TXT）解析有限元模型的物理属性及配置信息
    """
    @staticmethod
    def load_materials(file_path: str) -> Dict[int, Material]:
        """
        读取材料属性定义的 CSV 文件
        格式: id, E, rho
        """
        materials = {}
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # 跳过表头
            for row in reader:
                if not row or row[0].strip().startswith('#'):
                    continue
                mat_id = int(row[0].strip())
                E = float(row[1].strip())
                rho = float(row[2].strip())
                materials[mat_id] = Material(E=E, rho=rho)
        return materials

    @staticmethod
    def load_model(file_path: str, materials: Dict[int, Material]) -> Tuple[List[Node], List[Element2D], List[Tuple[int, int, float]], List[Tuple[int, int, float]], List[Tuple[int, int, float, float, float]], Dict[str, str]]:
        """
        从 TXT/CSV 文件加载有限元模型及求解配置。
        支持的模型指令：
        - NODE, id, x, y
        - ELEM, id, type, node1_id, node2_id, mat_id, section_A, section_I
        - BC, node_id, local_dof, value
        - LOAD, node_id, local_dof, value
        - DLOAD, node_id, local_dof, value, start_time, end_time
        - CONFIG, key, value
        """
        nodes_dict = {}
        elements = []
        bcs = []       # [(node_id, local_dof, value)]
        loads = []     # [(node_id, local_dof, value)]
        dloads = []    # [(node_id, local_dof, value, start_time, end_time)]
        configs = {}
        
        with open(file_path, mode='r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = [p.strip() for p in line.split(',')]
                cmd = parts[0].upper()
                
                if cmd == 'NODE':
                    nid, x, y = int(parts[1]), float(parts[2]), float(parts[3])
                    nodes_dict[nid] = Node(nid, x, y)
                elif cmd == 'ELEM':
                    eid, etype, n1, n2, mid = int(parts[1]), parts[2], int(parts[3]), int(parts[4]), int(parts[5])
                    sec_A, sec_I = float(parts[6]), float(parts[7])
                    sec = Section(A=sec_A, I=sec_I)
                    
                    if mid not in materials:
                        raise ValueError(f"Material ID {mid} not found for element {eid}")
                    mat = materials[mid]
                    
                    if etype.upper() == 'TRUSS2D':
                        el = TrussElement2D(eid, nodes_dict[n1], nodes_dict[n2], mat, sec)
                    elif etype.upper() == 'BEAM2D':
                        el = BeamElement2D(eid, nodes_dict[n1], nodes_dict[n2], mat, sec)
                    else:
                        raise ValueError(f"Unknown element type: {etype}")
                    elements.append(el)
                elif cmd == 'BC':
                    nid, dof, val = int(parts[1]), int(parts[2]), float(parts[3])
                    bcs.append((nid, dof, val))
                elif cmd == 'LOAD':
                    nid, dof, val = int(parts[1]), int(parts[2]), float(parts[3])
                    loads.append((nid, dof, val))
                elif cmd == 'DLOAD':
                    nid, dof, val, t_start, t_end = int(parts[1]), int(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    dloads.append((nid, dof, val, t_start, t_end))
                elif cmd == 'CONFIG':
                    key, val = parts[1], parts[2]
                    configs[key] = val
                    
        # 分配全局自由度
        nodes = list(nodes_dict.values())
        nodes.sort(key=lambda n: n.node_id)
        
        # 判断模型的自由度规格
        has_beam = any(isinstance(e, BeamElement2D) for e in elements)
        dofs_per_node = 3 if has_beam else 2
        
        dof_count = 0
        for n in nodes:
            n.dofs = list(range(dof_count, dof_count + dofs_per_node))
            dof_count += dofs_per_node
            
        return nodes, elements, bcs, loads, dloads, configs
