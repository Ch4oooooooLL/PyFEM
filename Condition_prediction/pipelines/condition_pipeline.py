from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(PACKAGE_DIR)
PYFEM_DIR = os.path.join(ROOT_DIR, "PyFEM_Dynamics")
for path in (ROOT_DIR, PYFEM_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from Condition_prediction.data.load_builder import build_deterministic_load_matrix
from Condition_prediction.inference.model_inference import ConditionDamagePredictor
from Condition_prediction.postprocess.comparison import (
    build_constant_damage_index_history,
    compute_fem_damage_index_history,
    evaluate_damage_history,
    plot_error_evolution,
    plot_focus_elements,
    plot_mean_damage_comparison,
    save_metrics,
)
from core.io_parser import YAMLParser
from pipeline.data_gen import run_fem_solver


def _resolve_path(base_dir: str, path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(base_dir, path_value))


def _apply_damage_case(elements: List[Any], damaged_elements: List[Dict[str, Any]]) -> np.ndarray:
    factors = np.ones(len(elements), dtype=np.float64)
    for item in damaged_elements:
        elem_id = int(item["element_id"])
        factor = float(item["factor"])
        if elem_id < 0 or elem_id >= len(elements):
            raise ValueError(f"element_id out of range: {elem_id}")
        if factor <= 0.0 or factor > 1.0:
            raise ValueError(f"factor must be in (0, 1], got {factor}")
        factors[elem_id] = factor
        elements[elem_id].material.E *= factor
    return factors


def _run_fem_pair(
    structure_file: str,
    dt: float,
    total_time: float,
    alpha: float,
    beta: float,
    load_specs: List[Dict[str, Any]],
    damaged_elements: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nodes_ref, elements_ref, bcs_ref = YAMLParser.build_structure_objects(structure_file)
    nodes_dmg, elements_dmg, bcs_dmg = YAMLParser.build_structure_objects(structure_file)

    num_nodes = len(nodes_ref)
    dofs_per_node = len(nodes_ref[0].dofs) if nodes_ref else 2
    time_axis, load_matrix = build_deterministic_load_matrix(
        load_specs=load_specs,
        num_nodes=num_nodes,
        dofs_per_node=dofs_per_node,
        dt=dt,
        total_time=total_time,
    )

    damage_factors = _apply_damage_case(elements_dmg, damaged_elements)

    disp_ref, stress_ref = run_fem_solver(
        nodes_ref,
        elements_ref,
        bcs_ref,
        load_matrix,
        dt,
        total_time,
        damping_alpha=alpha,
        damping_beta=beta,
    )
    disp_dmg, stress_dmg = run_fem_solver(
        nodes_dmg,
        elements_dmg,
        bcs_dmg,
        load_matrix,
        dt,
        total_time,
        damping_alpha=alpha,
        damping_beta=beta,
    )

    return time_axis, load_matrix, disp_ref, stress_ref, disp_dmg, stress_dmg, damage_factors


def run_condition_pipeline(config_path: str, output_dir_override: str | None = None) -> Dict[str, Any]:
    config_abs = os.path.abspath(config_path)
    if not os.path.exists(config_abs):
        raise FileNotFoundError(f"config not found: {config_abs}")

    config_dir = os.path.dirname(config_abs)
    with open(config_abs, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    structure_file = _resolve_path(config_dir, cfg.get("structure_file", "structure.yaml"))

    time_cfg = cfg.get("time", {})
    dt = float(time_cfg.get("dt", 0.01))
    total_time = float(time_cfg.get("total_time", 2.0))

    damping_cfg = cfg.get("damping", {})
    alpha = float(damping_cfg.get("alpha", 0.1))
    beta = float(damping_cfg.get("beta", 0.01))

    load_specs = cfg.get("load_case", {}).get("loads", [])
    if not load_specs:
        raise ValueError("load_case.loads is empty.")

    damaged_elements = cfg.get("damage_case", {}).get("damaged_elements", [])
    if not damaged_elements:
        raise ValueError("damage_case.damaged_elements is empty.")

    output_cfg = cfg.get("output", {})
    root_output_dir = output_dir_override or output_cfg.get("root_dir", "Condition_prediction/outputs")
    root_output_dir = _resolve_path(config_dir, root_output_dir)
    case_name = str(output_cfg.get("case_name", "condition_case"))
    focus_elements = output_cfg.get("focus_elements", [])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(root_output_dir, f"{case_name}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    time_axis, load_matrix, disp_ref, stress_ref, disp_dmg, stress_dmg, damage_factors = _run_fem_pair(
        structure_file=structure_file,
        dt=dt,
        total_time=total_time,
        alpha=alpha,
        beta=beta,
        load_specs=load_specs,
        damaged_elements=damaged_elements,
    )

    fem_damage_index = compute_fem_damage_index_history(stress_dmg, stress_ref)
    true_damage_index = build_constant_damage_index_history(
        num_steps=len(time_axis),
        damage_factors=damage_factors,
    )

    dl_cfg_raw = cfg.get("deep_learning", {})
    dl_cfg = dl_cfg_raw.get("inference", dl_cfg_raw)
    model_type = str(dl_cfg.get("model_type", "gt")).lower()
    feature_mode = str(dl_cfg.get("feature_mode", "response")).lower()
    checkpoint_path = _resolve_path(config_dir, dl_cfg.get("checkpoint", "Deep_learning/checkpoints/gt_best.pth"))
    dataset_npz = _resolve_path(config_dir, dl_cfg.get("dataset_npz", "dataset/train.npz"))
    device = str(dl_cfg.get("device", "auto"))

    predictor = ConditionDamagePredictor(
        structure_file=structure_file,
        dataset_npz=dataset_npz,
        checkpoint_path=checkpoint_path,
        model_type=model_type,  # type: ignore[arg-type]
        feature_mode=feature_mode,  # type: ignore[arg-type]
        device=device,
    )

    dl_damage_factor_history = predictor.predict_factor_history(disp_dmg, load_hist=load_matrix)
    dl_damage_index = np.clip(1.0 - dl_damage_factor_history, 0.0, 1.0)
    dl_damage_factor_final = predictor.predict_final_factor(disp_dmg, load_hist=load_matrix)

    metrics = evaluate_damage_history(
        predicted_damage_index=dl_damage_index,
        fem_damage_index=fem_damage_index,
        true_damage_index=true_damage_index,
    )

    npz_path = os.path.join(run_dir, "result_bundle.npz")
    np.savez_compressed(
        npz_path,
        time=time_axis,
        load=load_matrix,
        disp_reference=disp_ref,
        stress_reference=stress_ref,
        disp_damaged=disp_dmg,
        stress_damaged=stress_dmg,
        fem_damage_index=fem_damage_index,
        dl_damage_factor_history=dl_damage_factor_history,
        dl_damage_index=dl_damage_index,
        true_damage_factor=damage_factors,
        true_damage_index=true_damage_index,
        dl_damage_factor_final=dl_damage_factor_final,
    )

    plot_mean_damage_comparison(
        time_axis=time_axis,
        fem_damage_index=fem_damage_index,
        dl_damage_index=dl_damage_index,
        true_damage_index=true_damage_index,
        save_path=os.path.join(run_dir, "mean_damage_comparison.png"),
    )
    plot_error_evolution(
        time_axis=time_axis,
        fem_damage_index=fem_damage_index,
        dl_damage_index=dl_damage_index,
        true_damage_index=true_damage_index,
        save_path=os.path.join(run_dir, "error_evolution.png"),
    )
    plot_focus_elements(
        time_axis=time_axis,
        fem_damage_index=fem_damage_index,
        dl_damage_index=dl_damage_index,
        true_damage_index=true_damage_index,
        focus_elements=focus_elements,
        save_path=os.path.join(run_dir, "focus_elements_comparison.png"),
    )

    metrics_path = os.path.join(run_dir, "metrics.json")
    save_metrics(metrics, metrics_path)

    summary = {
        "config_path": config_abs,
        "run_dir": run_dir,
        "result_npz": npz_path,
        "metrics_path": metrics_path,
        "model_type": model_type,
        "feature_mode": feature_mode,
        "checkpoint": checkpoint_path,
        "metrics": metrics,
    }
    return summary
