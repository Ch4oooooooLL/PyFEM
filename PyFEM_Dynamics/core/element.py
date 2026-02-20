import numpy as np
from abc import ABC, abstractmethod
from core.node import Node
from core.material import Material
from core.section import Section

class Element2D(ABC):
    """
    2D 有限元单元基类。
    """
    def __init__(self, element_id: int, node1: Node, node2: Node, material: Material, section: Section):
        self.element_id: int = element_id
        self.node1: Node = node1
        self.node2: Node = node2
        self.material: Material = material
        self.section: Section = section

    @property
    def length(self) -> float:
        """计算并返回单元长度"""
        return float(np.hypot(self.node2.x - self.node1.x, self.node2.y - self.node1.y))

    @property
    def angle(self) -> float:
        """计算并返回单元与 X 轴的夹角 (弧度)"""
        return float(np.arctan2(self.node2.y - self.node1.y, self.node2.x - self.node1.x))

    @abstractmethod
    def get_local_stiffness(self) -> np.ndarray:
        """获取局部刚度矩阵"""
        pass

    @abstractmethod
    def get_local_mass(self, lumping: bool = False) -> np.ndarray:
        """获取局部质量矩阵"""
        pass

    @abstractmethod
    def get_transformation_matrix(self) -> np.ndarray:
        """获取从局部坐标系到全局坐标系的坐标变换矩阵"""
        pass

    def get_global_stiffness(self) -> np.ndarray:
        """计算并返回全局坐标系下的单元刚度矩阵 T^T * k * T"""
        k_local = self.get_local_stiffness()
        T = self.get_transformation_matrix()
        return T.T @ k_local @ T

    def get_global_mass(self, lumping: bool = False) -> np.ndarray:
        """计算并返回全局坐标系下的单元质量矩阵 T^T * m * T"""
        m_local = self.get_local_mass(lumping=lumping)
        T = self.get_transformation_matrix()
        return T.T @ m_local @ T

class TrussElement2D(Element2D):
    """
    2D 桁架单元 (每个节点 2 个自由度: u, v)
    """
    def get_local_stiffness(self) -> np.ndarray:
        E = self.material.E
        A = self.section.A
        L = self.length
        k = E * A / L
        return np.array([
            [ k,  0, -k,  0],
            [ 0,  0,  0,  0],
            [-k,  0,  k,  0],
            [ 0,  0,  0,  0]
        ], dtype=float)

    def get_local_mass(self, lumping: bool = False) -> np.ndarray:
        rho = self.material.rho
        A = self.section.A
        L = self.length
        m_total = rho * A * L
        
        if lumping:
            # 集中质量矩阵
            m = m_total / 2.0
            return np.diag([m, m, m, m])
        else:
            # 一致质量矩阵
            m = m_total / 6.0
            return np.array([
                [2, 0, 1, 0],
                [0, 2, 0, 1],
                [1, 0, 2, 0],
                [0, 1, 0, 2]
            ], dtype=float)

    def get_transformation_matrix(self) -> np.ndarray:
        """
        4x4 的坐标变换矩阵
        """
        c = np.cos(self.angle)
        s = np.sin(self.angle)
        return np.array([
            [ c,  s,  0,  0],
            [-s,  c,  0,  0],
            [ 0,  0,  c,  s],
            [ 0,  0, -s,  c]
        ], dtype=float)

class BeamElement2D(Element2D):
    """
    2D 梁单元 (每个节点 3 个自由度: u, v, theta), 基于 Euler-Bernoulli 梁理论
    """
    def get_local_stiffness(self) -> np.ndarray:
        E = self.material.E
        A = self.section.A
        I = self.section.I
        L = self.length
        
        k_axial = E * A / L
        k_v1 = 12 * E * I / (L**3)
        k_v2 = 6 * E * I / (L**2)
        k_v3 = 4 * E * I / L
        k_v4 = 2 * E * I / L
        
        return np.array([
            [ k_axial,  0,       0,      -k_axial,  0,       0     ],
            [ 0,        k_v1,    k_v2,    0,       -k_v1,    k_v2  ],
            [ 0,        k_v2,    k_v3,    0,       -k_v2,    k_v4  ],
            [-k_axial,  0,       0,       k_axial,  0,       0     ],
            [ 0,       -k_v1,   -k_v2,    0,        k_v1,   -k_v2  ],
            [ 0,        k_v2,    k_v4,    0,       -k_v2,    k_v3  ]
        ], dtype=float)

    def get_local_mass(self, lumping: bool = False) -> np.ndarray:
        rho = self.material.rho
        A = self.section.A
        L = self.length
        m_total = rho * A * L
        
        if lumping:
            # 集中质量矩阵 (仅适用于平动，转动质量有时被近似)
            m = m_total / 2.0
            m_rot = m_total * L**2 / 24.0
            return np.diag([m, m, m_rot, m, m, m_rot])
        else:
            # 一致质量矩阵
            coeff = m_total / 420.0
            M = np.zeros((6, 6), dtype=float)
            
            # 轴向部分
            M[np.ix_([0, 3], [0, 3])] = (m_total / 6.0) * np.array([
                [2, 1],
                [1, 2]
            ])
            
            # 弯曲部分
            M[1,1] = coeff * 156; M[1,2] = coeff * 22*L; M[1,4] = coeff * 54; M[1,5] = coeff * -13*L
            M[2,1] = M[1,2];      M[2,2] = coeff * 4*L**2; M[2,4] = coeff * 13*L; M[2,5] = coeff * -3*L**2
            M[4,1] = M[1,4];      M[4,2] = M[2,4];         M[4,4] = coeff * 156; M[4,5] = coeff * -22*L
            M[5,1] = M[1,5];      M[5,2] = M[2,5];         M[5,4] = M[4,5];      M[5,5] = coeff * 4*L**2
            
            return M

    def get_transformation_matrix(self) -> np.ndarray:
        """
        6x6 的坐标变换矩阵
        """
        c = np.cos(self.angle)
        s = np.sin(self.angle)
        
        T = np.zeros((6, 6), dtype=float)
        T[0:2, 0:2] = [[c, s], [-s, c]]
        T[2, 2] = 1.0
        T[3:5, 3:5] = [[c, s], [-s, c]]
        T[5, 5] = 1.0
        return T
