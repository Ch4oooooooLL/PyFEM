
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Tuple, List


class FEMDataset(Dataset):
    """
    FEM数据集加载器
    
    数据格式:
        load:   (N, T, num_dofs) - 载荷时程
        disp:   (N, T, num_dofs) - 位移响应
        stress: (N, T, num_elem) - 应力响应
        damage: (N, num_elem)    - 损伤程度 (1.0=无损)
    """
    
    def __init__(
        self,
        npz_path: str,
        mode: str = 'response',
        sensor_nodes: Optional[List[int]] = None,
        normalize: bool = True
    ):
        """
        Args:
            npz_path: NPZ数据文件路径
            mode: 数据模式
                - 'response': 使用位移响应预测损伤
                - 'load': 使用载荷预测损伤
                - 'both': 同时使用载荷和响应
            sensor_nodes: 传感器节点ID列表，None表示使用全部节点
            normalize: 是否归一化数据
        """
        data = np.load(npz_path)
        
        self.load = data['load']
        self.disp = data['disp']
        self.stress = data['stress']
        self.damage = data['damage']
        
        self.mode = mode
        self.num_samples = self.damage.shape[0]
        self.num_timesteps = self.disp.shape[1]
        self.num_dofs = self.disp.shape[2]
        self.num_elements = self.damage.shape[1]
        
        self.sensor_nodes = sensor_nodes
        self.normalize = normalize
        
        self._setup_sensors()
        
        if normalize:
            self._compute_normalization()
    
    def _setup_sensors(self):
        """设置传感器节点"""
        if self.sensor_nodes is not None:
            self.sensor_dofs = []
            for node_id in self.sensor_nodes:
                self.sensor_dofs.extend([node_id * 2, node_id * 2 + 1])
        else:
            self.sensor_dofs = list(range(self.num_dofs))
    
    def _compute_normalization(self):
        """计算归一化参数"""
        self.disp_mean = self.disp.mean(axis=(0, 1))
        self.disp_std = self.disp.std(axis=(0, 1)) + 1e-8
        self.load_mean = self.load.mean(axis=(0, 1))
        self.load_std = self.load.std(axis=(0, 1)) + 1e-8
        
        self.damage_mean = self.damage.mean(axis=0)
        self.damage_std = self.damage.std(axis=0) + 1e-8
    
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
        """
        Returns:
            x: 输入特征 (根据mode不同)
            y: 损伤标签 (num_elements,)
        """
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
    seed: int = 42
) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset, torch.utils.data.Subset]:
    """划分训练集、验证集、测试集"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)
    
    return train_set, val_set, test_set
