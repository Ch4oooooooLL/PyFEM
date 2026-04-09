import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Any, Dict, Optional, Tuple, List, Sequence


BASE_DATA_KEYS = {'load', 'disp', 'stress', 'damage', 'E'}


class FEMDataset(Dataset):
    """
    FEM dataset loader.

    Data format:
        load:   (N, T, num_dofs) - load history
        disp:   (N, T, num_dofs) - displacement response
        stress: (N, T, num_elem) - stress response
        damage: (N, num_elem)    - damage factor (1.0 = healthy)
    """

    def __init__(
        self,
        npz_path: str,
        mode: str = 'response',
        sensor_nodes: Optional[List[int]] = None,
        normalize: bool = True,
        normalization_stats: Optional[Dict[str, Any]] = None,
    ):
        with np.load(npz_path) as data:
            self.load = np.asarray(data['load'])
            self.disp = np.asarray(data['disp'])
            self.stress = np.asarray(data['stress'])
            self.damage = np.asarray(data['damage'])
            self.sample_metadata = self._extract_sample_metadata(data)

        self.mode = mode
        self.num_samples = self.damage.shape[0]
        self.num_timesteps = self.disp.shape[1]
        self.num_dofs = self.disp.shape[2]
        self.num_elements = self.damage.shape[1]

        self.sensor_nodes = sensor_nodes
        self.normalize = normalize

        self._setup_sensors()

        if normalize:
            if normalization_stats is not None:
                self.set_normalization_stats(normalization_stats)
            else:
                self.fit_normalization()

    def _extract_sample_metadata(self, data: Any) -> Dict[str, np.ndarray]:
        metadata: Dict[str, np.ndarray] = {}
        num_samples = int(np.asarray(data['damage']).shape[0])
        for key in data.files:
            if key in BASE_DATA_KEYS:
                continue
            value = np.asarray(data[key])
            if value.ndim >= 1 and value.shape[0] == num_samples:
                metadata[key] = value
        return metadata

    def _setup_sensors(self) -> None:
        if self.sensor_nodes is not None:
            self.sensor_dofs = []
            for node_id in self.sensor_nodes:
                self.sensor_dofs.extend([node_id * 2, node_id * 2 + 1])
        else:
            self.sensor_dofs = list(range(self.num_dofs))

    def fit_normalization(self, indices: Optional[Sequence[int]] = None) -> None:
        """Fit normalization statistics from the full dataset or a train-only subset."""
        if indices is None:
            disp_source = self.disp
            load_source = self.load
            damage_source = self.damage
        else:
            index_array = np.asarray(list(indices), dtype=int)
            if index_array.size == 0:
                raise ValueError("indices for normalization cannot be empty.")
            disp_source = self.disp[index_array]
            load_source = self.load[index_array]
            damage_source = self.damage[index_array]

        self.disp_mean = disp_source.mean(axis=(0, 1))
        self.disp_std = disp_source.std(axis=(0, 1)) + 1e-8
        self.load_mean = load_source.mean(axis=(0, 1))
        self.load_std = load_source.std(axis=(0, 1)) + 1e-8
        self.damage_mean = damage_source.mean(axis=0)
        self.damage_std = damage_source.std(axis=0) + 1e-8

    def set_normalization_stats(self, stats: Dict[str, Any]) -> None:
        self.disp_mean = np.asarray(stats['disp_mean'], dtype=np.float64)
        self.disp_std = np.asarray(stats['disp_std'], dtype=np.float64)
        self.load_mean = np.asarray(stats['load_mean'], dtype=np.float64)
        self.load_std = np.asarray(stats['load_std'], dtype=np.float64)

        damage_mean = stats.get('damage_mean')
        damage_std = stats.get('damage_std')
        self.damage_mean = (
            np.asarray(damage_mean, dtype=np.float64)
            if damage_mean is not None
            else self.damage.mean(axis=0)
        )
        self.damage_std = (
            np.asarray(damage_std, dtype=np.float64)
            if damage_std is not None
            else self.damage.std(axis=0) + 1e-8
        )

    def get_normalization_stats(self) -> Dict[str, np.ndarray]:
        return {
            'disp_mean': np.asarray(self.disp_mean, dtype=np.float64),
            'disp_std': np.asarray(self.disp_std, dtype=np.float64),
            'load_mean': np.asarray(self.load_mean, dtype=np.float64),
            'load_std': np.asarray(self.load_std, dtype=np.float64),
            'damage_mean': np.asarray(self.damage_mean, dtype=np.float64),
            'damage_std': np.asarray(self.damage_std, dtype=np.float64),
        }

    def get_sample_metadata(self, key: str, default: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        return self.sample_metadata.get(key, default)

    def get_damage_severity(self) -> np.ndarray:
        severity = self.sample_metadata.get('damage_severity')
        if severity is not None:
            return np.asarray(severity, dtype=np.float64)
        return 1.0 - np.min(self.damage, axis=1)

    def _normalize_disp(self, disp: np.ndarray) -> np.ndarray:
        if self.normalize:
            return (disp - self.disp_mean) / self.disp_std
        return disp

    def _normalize_load(self, load: np.ndarray) -> np.ndarray:
        if self.normalize:
            return (load - self.load_mean) / self.load_std
        return load

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == 'response':
            x = self._normalize_disp(self.disp[idx])
            x = x[:, self.sensor_dofs]
        elif self.mode == 'load':
            x = self._normalize_load(self.load[idx])
            x = x[:, self.sensor_dofs]
        elif self.mode == 'both':
            disp = self._normalize_disp(self.disp[idx])
            load = self._normalize_load(self.load[idx])
            x = np.concatenate([disp[:, self.sensor_dofs], load[:, self.sensor_dofs]], axis=-1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        y = self.damage[idx]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def split_dataset(
    dataset: FEMDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    strategy: str = 'random',
    damage_holdout_threshold: Optional[float] = None,
) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset, torch.utils.data.Subset]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    n = len(dataset)
    generator = torch.Generator().manual_seed(seed)

    if strategy == 'random':
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        indices = torch.randperm(n, generator=generator).tolist()

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
    elif strategy == 'damage_holdout':
        if damage_holdout_threshold is None:
            raise ValueError("damage_holdout_threshold is required for strategy='damage_holdout'.")

        severity = dataset.get_damage_severity()
        ood_indices = [int(i) for i in np.where(severity >= float(damage_holdout_threshold))[0]]
        ood_set = set(ood_indices)
        remaining = [i for i in range(n) if i not in ood_set]
        shuffled_remaining = [remaining[i] for i in torch.randperm(len(remaining), generator=generator).tolist()]

        n_train = int(len(remaining) * train_ratio)
        n_val = int(len(remaining) * val_ratio)
        n_id_test = int(len(remaining) * test_ratio)

        train_indices = shuffled_remaining[:n_train]
        val_indices = shuffled_remaining[n_train:n_train + n_val]
        id_test_indices = shuffled_remaining[n_train + n_val:n_train + n_val + n_id_test]
        test_indices = ood_indices + id_test_indices
    else:
        raise ValueError(f"Unsupported split strategy: {strategy}")

    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)
    return train_set, val_set, test_set
