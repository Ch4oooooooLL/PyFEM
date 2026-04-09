from __future__ import annotations

import os
from typing import Any, Dict, Literal, Optional

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(CURRENT_DIR)

from Deep_learning.models.gt_model import GTDamagePredictor
from Deep_learning.models.pinn_model import PINNDamagePredictor
from Deep_learning.models.pinn_model_v2 import PINNDamagePredictorV2
from PyFEM_Dynamics.core.io_parser import YAMLParser


def _infer_gt_hidden_dim(state_dict: Dict[str, torch.Tensor]) -> int:
    return int(state_dict["node_encoder.2.weight"].shape[0])


def _infer_gt_node_in_dim(state_dict: Dict[str, torch.Tensor]) -> int:
    return int(state_dict["node_encoder.0.weight"].shape[1])


def _infer_pinn_hidden_dim(state_dict: Dict[str, torch.Tensor]) -> int:
    return int(state_dict["encoder.0.weight"].shape[0])


def _infer_pinn_input_dim(state_dict: Dict[str, torch.Tensor]) -> int:
    return int(state_dict["encoder.0.weight"].shape[1])


def _infer_pinn_num_layers(state_dict: Dict[str, torch.Tensor]) -> int:
    return len([k for k in state_dict.keys() if k.startswith("encoder.") and k.endswith(".weight")])


class ConditionDamagePredictor:
    """
    Inference wrapper for trained Deep_learning models (GT / legacy PINN / PINN V2).
    """

    def __init__(
        self,
        structure_file: str,
        dataset_npz: str | None,
        checkpoint_path: str,
        model_type: Literal["gt", "legacy_pinn", "pinn", "pinn_v2"] = "gt",
        feature_mode: Literal["response", "load", "both"] = "response",
        device: str = "auto",
    ) -> None:
        self.model_type = str(model_type).lower()
        if self.model_type == "pinn":
            self.model_type = "legacy_pinn"
        if self.model_type not in {"gt", "legacy_pinn", "pinn_v2"}:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.feature_mode = feature_mode.lower()
        if self.feature_mode not in {"response", "load", "both"}:
            raise ValueError(f"Unsupported feature_mode: {feature_mode}")

        if not os.path.exists(structure_file):
            raise FileNotFoundError(f"structure_file not found: {structure_file}")
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

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.checkpoint_payload = checkpoint
            state_dict = checkpoint['model_state_dict']
        else:
            self.checkpoint_payload = {}
            state_dict = checkpoint

        self.training_task = str(self.checkpoint_payload.get('training_task', 'final_state')).lower()
        self.history_mode = 'causal_prefix' if self.training_task == 'online_prefix' else 'static_repeated_final'
        self.seq_len = int(self.checkpoint_payload.get('model_args', {}).get('seq_len', 0))

        self._load_preprocessing(dataset_npz)
        self.model = self._build_model(state_dict)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.adj = torch.from_numpy(self.adj_np).float().to(self.device)
        self.edge_index = torch.from_numpy(self.edge_index_np).long().to(self.device)

    def _load_preprocessing(self, dataset_npz: Optional[str]) -> None:
        preprocessing = self.checkpoint_payload.get('preprocessing')
        if preprocessing:
            self.disp_mean = np.asarray(preprocessing['disp_mean'], dtype=np.float64)
            self.disp_std = np.asarray(preprocessing['disp_std'], dtype=np.float64)
            self.load_mean = np.asarray(preprocessing['load_mean'], dtype=np.float64)
            self.load_std = np.asarray(preprocessing['load_std'], dtype=np.float64)
            if self.seq_len <= 0:
                self.seq_len = int(self.checkpoint_payload.get('model_args', {}).get('seq_len', 0))
            return

        if dataset_npz is None or not os.path.exists(dataset_npz):
            raise FileNotFoundError(
                "checkpoint is missing preprocessing statistics and dataset_npz is unavailable."
            )

        with np.load(dataset_npz) as data:
            disp = np.asarray(data["disp"], dtype=np.float64)
            load = np.asarray(data["load"], dtype=np.float64)
            self.seq_len = int(disp.shape[1])
            self.disp_mean = disp.mean(axis=(0, 1), dtype=np.float64)
            self.disp_std = disp.std(axis=(0, 1), dtype=np.float64) + 1e-8
            self.load_mean = load.mean(axis=(0, 1), dtype=np.float64)
            self.load_std = load.std(axis=(0, 1), dtype=np.float64) + 1e-8

    def _build_model(self, state_dict: Dict[str, torch.Tensor]) -> torch.nn.Module:
        model_args = dict(self.checkpoint_payload.get('model_args', {}))

        if self.model_type == "gt":
            if not model_args:
                model_args = {
                    'num_nodes': self.num_nodes,
                    'num_elements': self.num_elements,
                    'seq_len': self.seq_len,
                    'node_in_dim': _infer_gt_node_in_dim(state_dict),
                    'hidden_dim': _infer_gt_hidden_dim(state_dict),
                }
            self.expected_input_dim = int(model_args['num_nodes']) * int(model_args['node_in_dim'])
            return GTDamagePredictor(**model_args)

        if self.model_type == "legacy_pinn":
            if not model_args:
                model_args = {
                    'input_dim': _infer_pinn_input_dim(state_dict),
                    'hidden_dim': _infer_pinn_hidden_dim(state_dict),
                    'num_layers': _infer_pinn_num_layers(state_dict),
                    'output_dim': self.num_elements,
                }
            self.expected_input_dim = int(model_args['input_dim'])
            model = PINNDamagePredictor(**model_args)
            element_adjacency = np.zeros((self.num_elements, self.num_elements), dtype=np.float32)
            node_to_elements: Dict[int, list[int]] = {}
            for elem_idx, (n1, n2) in enumerate(self.edge_index_np):
                node_to_elements.setdefault(int(n1), []).append(elem_idx)
                node_to_elements.setdefault(int(n2), []).append(elem_idx)
            for elems in node_to_elements.values():
                for i in elems:
                    for j in elems:
                        if i != j:
                            element_adjacency[i, j] = 1.0
            model.set_structure_info(element_adjacency)
            return model

        if not model_args:
            raise ValueError("PINN V2 inference requires model_args in checkpoint.")
        self.expected_input_dim = int(model_args['input_dim'])
        model = PINNDamagePredictorV2(**model_args)
        element_adjacency = np.zeros((self.num_elements, self.num_elements), dtype=np.float32)
        node_to_elements: Dict[int, list[int]] = {}
        for elem_idx, (n1, n2) in enumerate(self.edge_index_np):
            node_to_elements.setdefault(int(n1), []).append(elem_idx)
            node_to_elements.setdefault(int(n2), []).append(elem_idx)
        for elems in node_to_elements.values():
            for i in elems:
                for j in elems:
                    if i != j:
                        element_adjacency[i, j] = 1.0
        model.set_structure_info(element_adjacency=element_adjacency)
        return model

    def _normalize_disp(self, disp_hist: np.ndarray) -> np.ndarray:
        disp_hist = np.asarray(disp_hist, dtype=np.float64)
        if disp_hist.ndim != 2:
            raise ValueError("disp_hist must be 2D (num_steps, num_dofs).")
        if disp_hist.shape[1] != self.num_dofs:
            raise ValueError(f"disp_hist has {disp_hist.shape[1]} dofs, expected {self.num_dofs}.")
        return (disp_hist - self.disp_mean) / self.disp_std

    def _normalize_load(self, load_hist: np.ndarray) -> np.ndarray:
        load_hist = np.asarray(load_hist, dtype=np.float64)
        if load_hist.ndim != 2:
            raise ValueError("load_hist must be 2D (num_steps, num_dofs).")
        if load_hist.shape[1] != self.num_dofs:
            raise ValueError(f"load_hist has {load_hist.shape[1]} dofs, expected {self.num_dofs}.")
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
                    f"Response input dim={disp_norm.shape[1]} does not match expected_input_dim={self.expected_input_dim}."
                )
            return disp_norm

        if load_hist is None:
            raise ValueError(f"feature_mode='{self.feature_mode}' requires load_hist.")
        load_norm = self._normalize_load(load_hist)
        if self.feature_mode == "load":
            if load_norm.shape[1] != self.expected_input_dim:
                raise ValueError(
                    f"Load input dim={load_norm.shape[1]} does not match expected_input_dim={self.expected_input_dim}."
                )
            return load_norm

        x = np.concatenate([disp_norm, load_norm], axis=-1)
        if x.shape[1] != self.expected_input_dim:
            raise ValueError(
                f"Built input dim={x.shape[1]} does not match expected_input_dim={self.expected_input_dim}."
            )
        return x

    @torch.no_grad()
    def predict_final_factor(self, disp_hist: np.ndarray, load_hist: np.ndarray | None = None) -> np.ndarray:
        x_norm = self._build_input_feature(disp_hist, load_hist=load_hist)
        x = torch.from_numpy(x_norm).float().unsqueeze(0).to(self.device)

        if self.model_type == "gt":
            pred = self.model(x, self.adj, self.edge_index)
        elif self.model_type == "legacy_pinn":
            pred = self.model(x)["damage"]
        else:
            pred = self.model(x)

        pred_np = pred.squeeze(0).detach().cpu().numpy().astype(np.float64)
        return np.clip(pred_np, 0.0, 1.0)

    @torch.no_grad()
    def predict_factor_history(
        self,
        disp_hist: np.ndarray,
        load_hist: np.ndarray | None = None,
    ) -> np.ndarray:
        x_norm = self._build_input_feature(disp_hist, load_hist=load_hist)
        num_steps = x_norm.shape[0]

        if self.training_task != 'online_prefix':
            final_factor = self.predict_final_factor(disp_hist, load_hist=load_hist)
            return np.repeat(final_factor[np.newaxis, :], repeats=num_steps, axis=0)

        predictions = np.zeros((num_steps, self.num_elements), dtype=np.float64)
        for step in range(num_steps):
            x_step = np.zeros_like(x_norm)
            x_step[: step + 1, :] = x_norm[: step + 1, :]
            x = torch.from_numpy(x_step).float().unsqueeze(0).to(self.device)

            if self.model_type == "gt":
                pred = self.model(x, self.adj, self.edge_index)
            elif self.model_type == "legacy_pinn":
                pred = self.model(x)["damage"]
            else:
                pred = self.model(x)

            predictions[step, :] = pred.squeeze(0).detach().cpu().numpy().astype(np.float64)

        return np.clip(predictions, 0.0, 1.0)
