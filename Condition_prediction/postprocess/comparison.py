from __future__ import annotations

import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def compute_fem_damage_index_history(
    stress_damaged: np.ndarray,
    stress_reference: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Convert stress history into a time-varying damage index.
    
    统一损伤指标定义：基于刚度折减的损伤指标
    由于应力与刚度成正比 (σ ∝ E^{-1} 对于相同载荷),
    应力比值可近似为刚度折减系数，即：
    damage_index = 1 - η ≈ 1 - |σ_damaged| / |σ_ref|
    
    此定义与深度学习模型输出的刚度因子 (damage_factor = η) 保持一致：
    - damage_index = 1 - η
    - 预测的损伤因子 η 对应结构刚度的保留比例

    shape: (num_steps, num_elements)
    """
    if stress_damaged.shape != stress_reference.shape:
        raise ValueError("stress_damaged and stress_reference must have same shape.")

    ref = np.abs(stress_reference)
    dmg = np.abs(stress_damaged)
    ratio = dmg / (ref + eps)
    return np.clip(1.0 - ratio, 0.0, 1.0)


def build_constant_damage_index_history(
    num_steps: int,
    damage_factors: np.ndarray,
) -> np.ndarray:
    """
    Build constant true-damage-index history from stiffness factors.
    """
    factor = np.asarray(damage_factors, dtype=np.float64)
    index = np.clip(1.0 - factor, 0.0, 1.0)
    return np.repeat(index[np.newaxis, :], repeats=int(num_steps), axis=0)


def evaluate_damage_history(
    predicted_damage_index: np.ndarray,
    fem_damage_index: np.ndarray,
    true_damage_index: np.ndarray,
) -> Dict[str, float]:
    if predicted_damage_index.shape != fem_damage_index.shape:
        raise ValueError("predicted_damage_index and fem_damage_index must have same shape.")
    if true_damage_index.shape != fem_damage_index.shape:
        raise ValueError("true_damage_index and fem_damage_index must have same shape.")

    err_vs_fem = predicted_damage_index - fem_damage_index
    err_vs_true = predicted_damage_index - true_damage_index

    metrics = {
        "time_mae_vs_fem": float(np.mean(np.abs(err_vs_fem))),
        "time_rmse_vs_fem": float(np.sqrt(np.mean(err_vs_fem**2))),
        "final_mae_vs_fem": float(np.mean(np.abs(err_vs_fem[-1]))),
        "final_rmse_vs_fem": float(np.sqrt(np.mean(err_vs_fem[-1] ** 2))),
        "time_mae_vs_true": float(np.mean(np.abs(err_vs_true))),
        "time_rmse_vs_true": float(np.sqrt(np.mean(err_vs_true**2))),
        "final_mae_vs_true": float(np.mean(np.abs(err_vs_true[-1]))),
        "final_rmse_vs_true": float(np.sqrt(np.mean(err_vs_true[-1] ** 2))),
    }
    return metrics


def _setup_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
        }
    )


def plot_mean_damage_comparison(
    time_axis: np.ndarray,
    fem_damage_index: np.ndarray,
    dl_damage_index: np.ndarray,
    true_damage_index: np.ndarray,
    save_path: str,
) -> None:
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(time_axis, np.mean(fem_damage_index, axis=1), label="FEM Damage Index", linewidth=2.0)
    ax.plot(time_axis, np.mean(dl_damage_index, axis=1), label="DL Damage Index", linewidth=2.0)
    ax.plot(time_axis, np.mean(true_damage_index, axis=1), label="True Damage Index", linewidth=1.8, linestyle=":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean Damage Index")
    ax.set_title("Mean Damage Evolution")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_error_evolution(
    time_axis: np.ndarray,
    fem_damage_index: np.ndarray,
    dl_damage_index: np.ndarray,
    true_damage_index: np.ndarray,
    save_path: str,
) -> None:
    _setup_plot_style()
    err_fem = np.sqrt(np.mean((dl_damage_index - fem_damage_index) ** 2, axis=1))
    err_true = np.sqrt(np.mean((dl_damage_index - true_damage_index) ** 2, axis=1))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(time_axis, err_fem, label="RMSE vs FEM", linewidth=2.0)
    ax.plot(time_axis, err_true, label="RMSE vs True", linewidth=2.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMSE")
    ax.set_title("Prediction Error Evolution")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_focus_elements(
    time_axis: np.ndarray,
    fem_damage_index: np.ndarray,
    dl_damage_index: np.ndarray,
    true_damage_index: np.ndarray,
    focus_elements: List[int],
    save_path: str,
) -> None:
    _setup_plot_style()

    num_elements = fem_damage_index.shape[1]
    valid_ids = [int(i) for i in focus_elements if 0 <= int(i) < num_elements]
    if not valid_ids:
        valid_ids = list(range(min(4, num_elements)))

    nrows = len(valid_ids)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(9, 2.5 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    for ax, elem_id in zip(axes, valid_ids):
        ax.plot(time_axis, fem_damage_index[:, elem_id], label="FEM", linewidth=2.0)
        ax.plot(time_axis, dl_damage_index[:, elem_id], label="DL", linewidth=2.0)
        ax.plot(time_axis, true_damage_index[:, elem_id], label="True", linewidth=1.8, linestyle=":")
        ax.set_ylabel(f"Elem {elem_id}")
        ax.legend(loc="best")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Element-wise Damage Evolution", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def save_metrics(metrics: Dict[str, float], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
