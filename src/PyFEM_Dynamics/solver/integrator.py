import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Tuple

class StaticSolver:
    """
    线性静力分析求解器
    """
    @staticmethod
    def solve(K: sp.csc_matrix, F: np.ndarray) -> np.ndarray:
        """
        求解 K * U = F
        """
        U = spla.spsolve(K, F)
        return U

class NewmarkBetaSolver:
    """
    Newmark-beta 隐式动力学时间积分求解器
    """
    def __init__(self, M: sp.csc_matrix, C: sp.csc_matrix, K: sp.csc_matrix, 
                 dt: float, total_T: float, gamma: float = 0.5, beta: float = 0.25):
        """
        :param M: 全局质量矩阵 (需处理过边界条件)
        :param C: 全局阻尼矩阵 (需处理过边界条件)
        :param K: 全局刚度矩阵 (需处理过边界条件)
        :param dt: 时间步长
        积分参数:
        - 平均加速度法(无条件稳定): gamma=0.5, beta=0.25
        - 线性加速度法(有条件稳定): gamma=0.5, beta=1/6
        """
        self.M = M
        self.C = C
        self.K = K
        self.dt = dt
        self.num_steps = int(total_T / dt) + 1
        self.gamma = gamma
        self.beta = beta
        
        self.total_dofs = K.shape[0]

    def solve(self, F_t: np.ndarray, U0: np.ndarray = None, V0: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行积分计算。
        :param F_t: 随时间变化的荷载矩阵，shape为 (total_dofs, num_steps)
        :param U0: 初始位移，默认为全 0
        :param V0: 初始速度，默认为全 0
        :return: (U, V, A)，shape 均为 (total_dofs, num_steps)
        """
        assert F_t.shape[1] >= self.num_steps, "载荷矩阵的时间步数需大于或等于计算总步数"
        
        if U0 is None:
            U0 = np.zeros(self.total_dofs)
        if V0 is None:
            V0 = np.zeros(self.total_dofs)
            
        # 初始化结果矩阵
        U = np.zeros((self.total_dofs, self.num_steps))
        V = np.zeros((self.total_dofs, self.num_steps))
        A = np.zeros((self.total_dofs, self.num_steps))
        
        U[:, 0] = U0
        V[:, 0] = V0
        
        # 计算初始加速度 A0: M*A0 = F0 - C*V0 - K*U0
        R0 = F_t[:, 0] - self.C.dot(V0) - self.K.dot(U0)
        A0 = spla.spsolve(self.M, R0)
        A[:, 0] = A0
        
        # Newmark 积分常数
        a0 = 1.0 / (self.beta * self.dt**2)
        a1 = self.gamma / (self.beta * self.dt)
        a2 = 1.0 / (self.beta * self.dt)
        a3 = 1.0 / (2.0 * self.beta) - 1.0
        a4 = self.gamma / self.beta - 1.0
        a5 = (self.dt / 2.0) * (self.gamma / self.beta - 2.0)
        a6 = self.dt * (1.0 - self.gamma)
        a7 = self.gamma * self.dt
        
        # 计算等效刚度矩阵 (K_hat = K + a0*M + a1*C)
        K_hat = self.K + a0 * self.M + a1 * self.C
        
        # 因式分解 K_hat (以加速每步的求解)
        # 注意: K_hat 通常为对称正定矩阵 (如使用 average acceleration) 且非奇异
        # 如果是极大型稀疏矩阵可考虑 spla.splu, 但 spsolve 已具备足够性能
        K_hat_lu = spla.splu(K_hat.tocsc())

        # 时间步迭代
        for i in range(1, self.num_steps):
            u_prev = U[:, i-1]
            v_prev = V[:, i-1]
            a_prev = A[:, i-1]
            
            # 计算等效荷载块:
            # F_hat_i = F_i + M*(a0*u_prev + a2*v_prev + a3*a_prev) + C*(a1*u_prev + a4*v_prev + a5*a_prev)
            term_M = a0 * u_prev + a2 * v_prev + a3 * a_prev
            term_C = a1 * u_prev + a4 * v_prev + a5 * a_prev
            
            F_hat = F_t[:, i] + self.M.dot(term_M) + self.C.dot(term_C)
            
            # 求解 U_i
            u_next = K_hat_lu.solve(F_hat)
            
            # 更新加速度和速度
            a_next = a0 * (u_next - u_prev) - a2 * v_prev - a3 * a_prev
            v_next = v_prev + a6 * a_prev + a7 * a_next
            
            U[:, i] = u_next
            V[:, i] = v_next
            A[:, i] = a_next
            
        return U, V, A

    @staticmethod
    def compute_rayleigh_damping(M: sp.csc_matrix, K: sp.csc_matrix, alpha: float, beta: float) -> sp.csc_matrix:
        """
        计算 Rayleigh 阻尼矩阵: C = alpha * M + beta * K
        """
        return alpha * M + beta * K
