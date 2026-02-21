import numpy as np
from typing import List, Tuple

from core.element import TrussElement2D


def recover_truss_stress_time_history(
    elements: List[TrussElement2D], U_hist: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于全时程位移恢复桁架单元轴向应力和等效应力(von Mises近似)。

    :param elements: Truss2D 单元列表
    :param U_hist: 位移时程, shape=(total_dofs, num_steps)
    :return: (sigma_axial, sigma_vm), shape 均为 (num_steps, num_elements)
    """
    if U_hist.ndim != 2:
        raise ValueError("U_hist must be 2D array with shape (total_dofs, num_steps)")

    for element in elements:
        if not isinstance(element, TrussElement2D):
            raise NotImplementedError(
                "Stress recovery currently supports TrussElement2D only"
            )

    num_steps = U_hist.shape[1]
    num_elements = len(elements)

    sigma_axial = np.zeros((num_steps, num_elements), dtype=float)
    sigma_vm = np.zeros((num_steps, num_elements), dtype=float)

    for elem_idx, element in enumerate(elements):
        e_mod = element.material.E
        length = element.length
        transform = element.get_transformation_matrix()

        if length <= 0.0:
            raise ValueError(f"Element {element.element_id} has non-positive length")

        dofs = [
            element.node1.dofs[0],
            element.node1.dofs[1],
            element.node2.dofs[0],
            element.node2.dofs[1],
        ]

        for step_idx in range(num_steps):
            u_global = U_hist[dofs, step_idx]
            u_local = transform @ u_global
            epsilon = (u_local[2] - u_local[0]) / length
            sigma = e_mod * epsilon

            sigma_axial[step_idx, elem_idx] = sigma
            sigma_vm[step_idx, elem_idx] = abs(sigma)

    return sigma_axial, sigma_vm
