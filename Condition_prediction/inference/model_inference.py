from __future__ import annotations

import os
import sys
from typing import Dict, Literal

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(PACKAGE_DIR)
PYFEM_DIR = os.path.join(ROOT_DIR, "PyFEM_Dynamics")
for path in (ROOT_DIR, PYFEM_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from Deep_learning.models.gt_model import GTDamagePredictor
from Deep_learning.models.pinn_model import PINNDamagePredictor
from core.io_parser import YAMLParser


def _infer_gt_hidden_dim(state_dict: Dict[str, torch.Tensor]) -> int:
    key = "node_encoder.2.weight"
    if key not in state_dict:
        raise KeyError(f"Cannot infer GT hidden_dim. Missing key: {key}")
    return int(state_dict[key].shape[0])


def _infer_gt_node_in_dim(state_dict: Dict[str, torch.Tensor]) -> int:
    key = "node_encoder.0.weight"
    if key not in state_dict:
        raise KeyError(f"Cannot infer GT node_in_dim. Missing key: {key}")
    return int(state_dict[key].shape[1])


def _infer_pinn_hidden_dim(state_dict: Dict[str, torch.Tensor]) -> int:
    key = "encoder.0.weight"
    if key not in state_dict:
        raise KeyError(f"Cannot infer PINN hidden_dim. Missing key: {key}")
    return int(state_dict[key].shape[0])


def _infer_pinn_input_dim(state_dict: Dict[str, torch.Tensor]) -> int:
    key = "encoder.0.weight"
    if key not in state_dict:
        raise KeyError(f"Cannot infer PINN input_dim. Missing key: {key}")
    return int(state_dict[key].shape[1])


def _infer_pinn_num_layers(state_dict: Dict[str, torch.Tensor]) -> int:
    keys = [k for k in state_dict.keys() if k.startswith("encoder.") and k.endswith(".weight")]
    return len(keys)


class ConditionDamagePredictor:
    """
    Inference wrapper for trained Deep_learning models (GT/PINN).
    """

    def __init__(
        self,
        structure_file: str,
        dataset_npz: str,
        checkpoint_path: str,
        model_type: Literal["gt", "pinn"] = "gt",
        feature_mode: Literal["response", "load", "both"] = "response",
        device: str = "auto",
    ) -> None:
        self.model_type = model_type.lower()
        if self.model_type not in {"gt", "pinn"}:
            raise ValueError(f"Unsupported model_type: {model_type}")
        self.feature_mode = feature_mode.lower()
        if self.feature_mode not in {"response", "load", "both"}:
            raise ValueError(f"Unsupported feature_mode: {feature_mode}")

        if not os.path.exists(structure_file):
            raise FileNotFoundError(f"structure_file not found: {structure_file}")
        if not os.path.exists(dataset_npz):
            raise FileNotFoundError(f"dataset_npz not found: {dataset_npz}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        structure_data = YAMLParser.load_structure_yaml(structure_file)
        self.num_nodes = int(structure_data.num_nodes)
        self.num_elements = int(structure_data.num_elements)
        self.num_dofs = int(structure_data.num_dofs)
        self.dofs_per_node = int(structure_data.dofs_per_node)
        self.edge_index_np = np.asarray(structure_data.element_conn, dtype=np.int64)
        self.adj_np = np.eye(self.num_nodes, dtype=np.float32)
        for n1, n2 in self.edge_index_np:
            self.adj_np[n1, n2] = 1.0
            self.adj_np[n2, n1] = 1.0

        with np.load(dataset_npz) as data:
            disp = np.asarray(data["disp"], dtype=np.float64)
            load = np.asarray(data["load"], dtype=np.float64)
            self.seq_len = int(disp.shape[1])
            self.disp_mean = disp.mean(axis=(0, 1), dtype=np.float64)
            self.disp_std = disp.std(axis=(0, 1), dtype=np.float64) + 1e-8
            self.load_mean = load.mean(axis=(0, 1), dtype=np.float64)
            self.load_std = load.std(axis=(0, 1), dtype=np.float64) + 1e-8

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        if self.model_type == "gt":
            hidden_dim = _infer_gt_hidden_dim(state_dict)
            node_in_dim = _infer_gt_node_in_dim(state_dict)
            model = GTDamagePredictor(
                num_nodes=self.num_nodes,
                num_elements=self.num_elements,
                seq_len=self.seq_len,
                node_in_dim=node_in_dim,
                hidden_dim=hidden_dim,
            )
            self.expected_input_dim = int(self.num_nodes * node_in_dim)
        else:
            hidden_dim = _infer_pinn_hidden_dim(state_dict)
            input_dim = _infer_pinn_input_dim(state_dict)
            num_layers = _infer_pinn_num_layers(state_dict)
            model = PINNDamagePredictor(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=self.num_elements,
            )
            self.expected_input_dim = int(input_dim)
            element_adjacency = np.zeros((self.num_elements, self.num_elements), dtype=np.float32)
            node_to_elements = {}
            for elem_idx, (n1, n2) in enumerate(self.edge_index_np):
                node_to_elements.setdefault(int(n1), []).append(elem_idx)
                node_to_elements.setdefault(int(n2), []).append(elem_idx)
            for elems in node_to_elements.values():
                for i in elems:
                    for j in elems:
                        if i != j:
                            element_adjacency[i, j] = 1.0
            model.set_structure_info(element_adjacency)

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model

        self.adj = torch.from_numpy(self.adj_np).float().to(self.device)
        self.edge_index = torch.from_numpy(self.edge_index_np).long().to(self.device)

    def _normalize_disp(self, disp_hist: np.ndarray) -> np.ndarray:
        disp_hist = np.asarray(disp_hist, dtype=np.float64)
        if disp_hist.ndim != 2:
            raise ValueError("disp_hist must be 2D (num_steps, num_dofs).")
        if disp_hist.shape[1] != self.num_dofs:
            raise ValueError(
                f"disp_hist has {disp_hist.shape[1]} dofs, expected {self.num_dofs}."
            )
        return (disp_hist - self.disp_mean) / self.disp_std

    def _normalize_load(self, load_hist: np.ndarray) -> np.ndarray:
        load_hist = np.asarray(load_hist, dtype=np.float64)
        if load_hist.ndim != 2:
            raise ValueError("load_hist must be 2D (num_steps, num_dofs).")
        if load_hist.shape[1] != self.num_dofs:
            raise ValueError(
                f"load_hist has {load_hist.shape[1]} dofs, expected {self.num_dofs}."
            )
        return (load_hist - self.load_mean) / self.load_std

    def _build_input_feature(
        self,
        disp_hist: np.ndarray,
        load_hist: np.ndarray | None = None,
    ) -> np.ndarray:
        disp_norm = self._normalize_disp(disp_hist)
        if self.feature_mode == "response":
            if disp_norm.shape[1] != self.expected_input_dim:
                raise ValueError(
                    f"Response input dim={disp_norm.shape[1]} does not match model expected_input_dim="
                    f"{self.expected_input_dim}. Check feature_mode and training dataset mode."
                )
            return disp_norm

        if load_hist is None:
            raise ValueError(f"feature_mode='{self.feature_mode}' requires load_hist.")
        load_norm = self._normalize_load(load_hist)
        if self.feature_mode == "load":
            if load_norm.shape[1] != self.expected_input_dim:
                raise ValueError(
                    f"Load input dim={load_norm.shape[1]} does not match model expected_input_dim="
                    f"{self.expected_input_dim}. Check feature_mode and training dataset mode."
                )
            return load_norm

        x = np.concatenate([disp_norm, load_norm], axis=-1)
        if x.shape[1] != self.expected_input_dim:
            raise ValueError(
                f"Built input dim={x.shape[1]} does not match model expected_input_dim={self.expected_input_dim}. "
                f"Check feature_mode and training dataset mode."
            )
        return x

    @torch.no_grad()
    def predict_final_factor(self, disp_hist: np.ndarray, load_hist: np.ndarray | None = None) -> np.ndarray:
        """
        Predict final stiffness factor per element (1.0 means undamaged).
        """
        x_norm = self._build_input_feature(disp_hist, load_hist=load_hist)
        x = torch.from_numpy(x_norm).float().unsqueeze(0).to(self.device)

        if self.model_type == "gt":
            pred = self.model(x, self.adj, self.edge_index)
        else:
            pred = self.model(x)["damage"]

        pred_np = pred.squeeze(0).detach().cpu().numpy().astype(np.float64)
        return np.clip(pred_np, 0.0, 1.0)

    @torch.no_grad()
    def predict_factor_history(
        self,
        disp_hist: np.ndarray,
        load_hist: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Predict evolving stiffness factor by prefix-observation.

        At step k, only [0..k] response is visible and future steps are zero-padded.
        Returns:
            shape=(num_steps, num_elements), each value in [0, 1].
        """
        x_norm = self._build_input_feature(disp_hist, load_hist=load_hist)
        num_steps = x_norm.shape[0]
        predictions = np.zeros((num_steps, self.num_elements), dtype=np.float64)

        for step in range(num_steps):
            x_step = np.zeros_like(x_norm)
            x_step[: step + 1, :] = x_norm[: step + 1, :]
            x = torch.from_numpy(x_step).float().unsqueeze(0).to(self.device)

            if self.model_type == "gt":
                pred = self.model(x, self.adj, self.edge_index)
            else:
                pred = self.model(x)["damage"]

            predictions[step, :] = pred.squeeze(0).detach().cpu().numpy().astype(np.float64)

        return np.clip(predictions, 0.0, 1.0)
