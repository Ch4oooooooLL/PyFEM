import numpy as np
import scipy.sparse as sp
from typing import List
from core.element import Element2D

class Assembler:
    """
    负责组装全局刚度矩阵和质量矩阵。
    """
    def __init__(self, elements: List[Element2D], total_dofs: int):
        """
        :param elements: 模型中所有单元的列表
        :param total_dofs: 模型的总自由度数量
        """
        self.elements = elements
        self.total_dofs = total_dofs

    def assemble_K(self) -> sp.csc_matrix:
        """
        组装全局刚度矩阵 K
        推荐使用 lil_matrix 组装，最终返回 csc_matrix 或 csr_matrix。
        """
        K_global = sp.lil_matrix((self.total_dofs, self.total_dofs), dtype=float)
        
        for element in self.elements:
            k_global_element = element.get_global_stiffness()
            # 获取节点对应的全局自由度
            dofs = element.node1.dofs + element.node2.dofs
            
            # 使用 np.ix_ 构建切片索引进行矩阵块累加
            K_global[np.ix_(dofs, dofs)] += k_global_element
            
        return K_global.tocsc()

    def assemble_M(self, lumping: bool = False) -> sp.csc_matrix:
        """
        组装全局质量矩阵 M
        """
        M_global = sp.lil_matrix((self.total_dofs, self.total_dofs), dtype=float)
        
        for element in self.elements:
            m_global_element = element.get_global_mass(lumping=lumping)
            dofs = element.node1.dofs + element.node2.dofs
            
            M_global[np.ix_(dofs, dofs)] += m_global_element
            
        return M_global.tocsc()
