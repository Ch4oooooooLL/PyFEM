import numpy as np
import scipy.sparse as sp
from typing import List, Tuple

class BoundaryCondition:
    """
    处理模型边界条件，防止求解出现奇异矩阵。
    实现了乘大数法 (Penalty Method) 和矩阵划零划一法。
    """
    def __init__(self):
        # 存储本质边界条件: (dof_index, value)
        self.dirichlet_bcs: List[Tuple[int, float]] = []

    def add_dirichlet_bc(self, dof_index: int, value: float = 0.0):
        """
        添加位移约束（本质边界条件）
        :param dof_index: 约束的全局自由度编号
        :param value: 约束的给定值 (通常为 0)
        """
        self.dirichlet_bcs.append((dof_index, value))

    def apply_penalty_method(self, K: sp.csc_matrix, F: np.ndarray, penalty: float = 1e15) -> Tuple[sp.csc_matrix, np.ndarray]:
        """
        使用乘大数法 (罚函数法) 施加边界条件。
        此方法修改 K 矩阵的对角线元素并更新对应的荷载 F，无需改变矩阵维度且不会出现奇异。
        
        :param K: 未处理的全局刚度矩阵
        :param F: 未处理的全局外力向量
        :param penalty: 极大的惩罚因子，默认 1e15
        :return: 处理后的 (K_mod, F_mod)
        """
        # 将 K 转换为 lil 格式方便修改对角线
        K_mod = K.tolil()
        F_mod = F.copy()
        
        for dof, val in self.dirichlet_bcs:
            K_mod[dof, dof] *= penalty
            F_mod[dof] = K_mod[dof, dof] * val
            
        return K_mod.tocsc(), F_mod

    def apply_zero_one_method(self, K: sp.csc_matrix, F: np.ndarray) -> Tuple[sp.csc_matrix, np.ndarray]:
        """
        使用矩阵划零划一法施加边界条件。
        严格确保边界条件自由度对应的行和列全为 0，只有对角线为 1。
        
        :param K: 未处理的全局刚度矩阵
        :param F: 未处理的全局外力向量
        :return: 处理后的 (K_mod, F_mod)
        """
        # 转化为 LIL 以进行行列操作
        K_mod = K.tolil()
        F_mod = F.copy()
        
        for dof, val in self.dirichlet_bcs:
            # 修改与由于约束导致的力修正（提取列并处理非零边界值的影响）
            # P_i = P_i - K_ij * U_j
            if val != 0.0:
                for i in range(K_mod.shape[0]):
                    if i != dof:
                        F_mod[i] -= K_mod[i, dof] * val
            
            # 对应的行和列设为 0
            # lil_matrix 能够很方便地对行进行赋值，由于是稀疏矩阵不能直接切片赋值列
            # 所以我们将对应的非零元显式地置为 0
            row_nonzeros = K_mod.rows[dof]
            for col in row_nonzeros:
                K_mod[dof, col] = 0.0
            
            # 列操作：需要遍历所有行
            # 这里优化一点的做法是使用 CSC 或者 CSR 但为了通用直接用简单的 LIL 遍历。
            # 更好的实现是用 CSR/CSC 的操作或者使用 lil_matrix 转换来解决。
            pass  # 在下一步中通过特定方式完成对列的划0
            
        # 更高效的划零划一法实现：
        # 对于所有受约束的 DOF
        bc_dofs = [bc[0] for bc in self.dirichlet_bcs]
        if not bc_dofs:
            return K_mod.tocsc(), F_mod

        # 在 CSC/CSR 中清空约束所对应的行和列更快捷
        K_mod = K_mod.tocsr()
        
        # 为了清空列，可以将其转换为 csc
        K_csc = K_mod.tocsc()
        for dof in bc_dofs:
            K_csc.data[K_csc.indptr[dof]:K_csc.indptr[dof+1]] = 0.0
            
        # 转换回 LIL 清空行并设置对角线
        K_lil = K_csc.tolil()
        for dof, val in self.dirichlet_bcs:
            K_lil.rows[dof] = [dof]
            K_lil.data[dof] = [1.0]
            F_mod[dof] = val
            
        return K_lil.tocsc(), F_mod

    def apply_to_M(self, M: sp.csc_matrix) -> sp.csc_matrix:
        """
        在进行动力学分析时，M 矩阵也需要对应处理（通常边界点的加速度为0）。
        将边界约束点所在行列清0，对角线置1。
        """
        if not self.dirichlet_bcs:
            return M
            
        bc_dofs = [bc[0] for bc in self.dirichlet_bcs]
        M_csc = M.tocsc()
        
        for dof in bc_dofs:
            M_csc.data[M_csc.indptr[dof]:M_csc.indptr[dof+1]] = 0.0
            
        M_lil = M_csc.tolil()
        for dof in bc_dofs:
            M_lil.rows[dof] = [dof]
            M_lil.data[dof] = [1.0]
            
        return M_lil.tocsc()
