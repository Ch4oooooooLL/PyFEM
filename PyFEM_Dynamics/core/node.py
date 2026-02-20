from typing import List

class Node:
    """
    表示 2D 有限元模型中的一个节点。
    """
    def __init__(self, node_id: int, x: float, y: float):
        self.node_id: int = node_id
        self.x: float = x
        self.y: float = y
        self.dofs: List[int] = []  # 分配给该节点的全局自由度编号列表

    def __repr__(self) -> str:
        return f"Node(id={self.node_id}, x={self.x}, y={self.y}, dofs={self.dofs})"
