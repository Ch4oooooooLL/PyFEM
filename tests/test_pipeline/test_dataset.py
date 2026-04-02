"""
Tests for FEMDataset.
"""

import pytest
import numpy as np
import tempfile
import os

# Skip all tests if torch is not available
torch = pytest.importorskip("torch")

from Deep_learning.data.dataset import FEMDataset, split_dataset
from PyFEM_Dynamics.pipeline.data_gen import apply_damage
from PyFEM_Dynamics.core.io_parser import YAMLParser


class TestFEMDataset:
    """Test cases for FEMDataset."""
    
    @pytest.fixture
    def sample_npz_data(self):
        """Create sample NPZ data for testing."""
        np.random.seed(42)
        n_samples = 100
        n_timesteps = 50
        n_dofs = 20
        n_elements = 15
        
        data = {
            'load': np.random.randn(n_samples, n_timesteps, n_dofs).astype(np.float32),
            'disp': np.random.randn(n_samples, n_timesteps, n_dofs).astype(np.float32),
            'stress': np.random.randn(n_samples, n_timesteps, n_elements).astype(np.float32),
            'damage': np.random.rand(n_samples, n_elements).astype(np.float32)
        }
        return data
    
    @pytest.fixture
    def temp_npz_file(self, sample_npz_data):
        """Create temporary NPZ file."""
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            np.savez(f.name, **sample_npz_data)
            yield f.name
        os.unlink(f.name)
    
    def test_dataset_creation(self, temp_npz_file):
        """Test dataset can be loaded from NPZ."""
        dataset = FEMDataset(temp_npz_file, mode='response', normalize=False)
        
        assert dataset.num_samples == 100
        assert dataset.num_timesteps == 50
        assert dataset.num_dofs == 20
        assert dataset.num_elements == 15
    
    def test_dataset_length(self, temp_npz_file):
        """Test dataset length is correct."""
        dataset = FEMDataset(temp_npz_file, mode='response', normalize=False)
        assert len(dataset) == 100
    
    def test_getitem_response_mode(self, temp_npz_file):
        """Test getting item in response mode."""
        dataset = FEMDataset(temp_npz_file, mode='response', normalize=False)
        
        x, y = dataset[0]
        
        # x should be disp with shape (timesteps, dofs)
        assert x.shape == (50, 20)
        # y should be damage with shape (num_elements,)
        assert y.shape == (15,)
    
    def test_getitem_load_mode(self, temp_npz_file):
        """Test getting item in load mode."""
        dataset = FEMDataset(temp_npz_file, mode='load', normalize=False)
        
        x, y = dataset[0]
        
        # x should be load with shape (timesteps, dofs)
        assert x.shape == (50, 20)
        assert y.shape == (15,)
    
    def test_normalization(self, temp_npz_file):
        """Test data normalization."""
        dataset_norm = FEMDataset(temp_npz_file, mode='response', normalize=True)
        dataset_raw = FEMDataset(temp_npz_file, mode='response', normalize=False)
        
        x_norm, _ = dataset_norm[0]
        x_raw, _ = dataset_raw[0]
        
        # Normalized data should have different values
        assert not torch.allclose(x_norm, x_raw)
    
    def test_sensor_nodes(self, temp_npz_file):
        """Test selecting specific sensor nodes."""
        sensor_nodes = [0, 1, 2]  # Use first 3 nodes (6 DOFs)
        dataset = FEMDataset(temp_npz_file, mode='response', sensor_nodes=sensor_nodes, normalize=False)
        
        x, y = dataset[0]
        
        # Should have 6 DOFs (3 nodes * 2 DOFs per node)
        assert x.shape == (50, 6)

    def test_fit_normalization_uses_train_indices_only(self, sample_npz_data):
        """Normalization statistics should be fit from the designated training subset only."""
        sample_npz_data['disp'][:50] = 1.0
        sample_npz_data['disp'][50:] = 101.0

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            np.savez(f.name, **sample_npz_data)
            tmp_path = f.name

        try:
            dataset = FEMDataset(tmp_path, mode='response', normalize=True)
            dataset.fit_normalization(indices=list(range(50)))

            assert np.allclose(dataset.disp_mean, 1.0)
            assert np.allclose(dataset.disp_std, 1e-8)
        finally:
            os.unlink(tmp_path)


class TestSplitDataset:
    """Test cases for dataset splitting."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a small dataset for splitting tests."""
        np.random.seed(42)
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            data = {
                'load': np.random.randn(100, 10, 6).astype(np.float32),
                'disp': np.random.randn(100, 10, 6).astype(np.float32),
                'stress': np.random.randn(100, 10, 5).astype(np.float32),
                'damage': np.random.rand(100, 5).astype(np.float32)
            }
            np.savez(f.name, **data)
            dataset = FEMDataset(f.name, mode='response', normalize=False)
            yield dataset
        os.unlink(f.name)
    
    def test_split_sizes(self, sample_dataset):
        """Test dataset split produces correct sizes."""
        train_set, val_set, test_set = split_dataset(
            sample_dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
        )
        
        assert len(train_set) == 70
        assert len(val_set) == 20
        assert len(test_set) == 10
    
    def test_split_deterministic(self, sample_dataset):
        """Test split is deterministic with same seed."""
        train1, val1, test1 = split_dataset(sample_dataset, seed=42)
        train2, val2, test2 = split_dataset(sample_dataset, seed=42)
        
        # Get indices
        idx1_train = train1.indices
        idx2_train = train2.indices
        
        assert idx1_train == idx2_train

    def test_damage_holdout_split_reserves_severe_samples_for_test(self):
        """OOD split should reserve severe damage samples for the test subset."""
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            damage = np.ones((10, 3), dtype=np.float32)
            damage[:3, 0] = np.array([0.2, 0.25, 0.3], dtype=np.float32)
            data = {
                'load': np.zeros((10, 4, 6), dtype=np.float32),
                'disp': np.zeros((10, 4, 6), dtype=np.float32),
                'stress': np.zeros((10, 4, 3), dtype=np.float32),
                'damage': damage,
            }
            np.savez(f.name, **data)
            dataset = FEMDataset(f.name, mode='response', normalize=False)
        try:
            train_set, val_set, test_set = split_dataset(
                dataset,
                train_ratio=0.7,
                val_ratio=0.3,
                test_ratio=0.0,
                seed=42,
                strategy='damage_holdout',
                damage_holdout_threshold=0.65,
            )

            assert len(test_set) == 3
            assert set(test_set.indices) == {0, 1, 2}
            assert set(train_set.indices).isdisjoint(test_set.indices)
            assert set(val_set.indices).isdisjoint(test_set.indices)
        finally:
            os.unlink(f.name)


class TestDamageGeneration:
    """Test cases for configurable damage generation."""

    def test_apply_damage_can_generate_healthy_sample(self):
        """Damage generation should support explicit healthy samples."""
        root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
        structure_file = os.path.abspath(os.path.join(root_dir, 'structure.yaml'))
        _, elements, _ = YAMLParser.build_structure_objects(structure_file)
        rng = np.random.default_rng(42)

        damage, damaged_e = apply_damage(
            elements,
            {
                'enabled': True,
                'healthy_sample_ratio': 1.0,
                'min_damaged_elements': 1,
                'max_damaged_elements': 3,
                'reduction_range': [0.5, 0.9],
            },
            rng,
        )

        assert np.allclose(damage, 1.0)
        assert np.allclose(damaged_e, [el.material.E for el in elements])
