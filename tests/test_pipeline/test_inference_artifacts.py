import os
import tempfile

import numpy as np
import pytest
import torch

from Condition_prediction.inference.model_inference import ConditionDamagePredictor
from Deep_learning.models.gt_model import GTDamagePredictor
from core.io_parser import YAMLParser


@pytest.fixture
def structure_file() -> str:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    return os.path.join(root_dir, 'structure.yaml')


def test_condition_predictor_prefers_checkpoint_preprocessing(structure_file):
    """Inference should load preprocessing stats from checkpoint artifacts without reopening the dataset."""
    structure_data = YAMLParser.load_structure_yaml(structure_file)
    num_nodes = int(structure_data.num_nodes)
    num_elements = int(structure_data.num_elements)
    seq_len = 4
    node_in_dim = 2

    model = GTDamagePredictor(
        num_nodes=num_nodes,
        num_elements=num_elements,
        seq_len=seq_len,
        node_in_dim=node_in_dim,
        hidden_dim=16,
    )

    checkpoint_payload = {
        'model_state_dict': model.state_dict(),
        'model_name': 'gt',
        'model_args': {
            'num_nodes': num_nodes,
            'num_elements': num_elements,
            'seq_len': seq_len,
            'node_in_dim': node_in_dim,
            'hidden_dim': 16,
        },
        'preprocessing': {
            'disp_mean': np.zeros(num_nodes * node_in_dim, dtype=np.float64).tolist(),
            'disp_std': np.ones(num_nodes * node_in_dim, dtype=np.float64).tolist(),
            'load_mean': np.zeros(num_nodes * node_in_dim, dtype=np.float64).tolist(),
            'load_std': np.ones(num_nodes * node_in_dim, dtype=np.float64).tolist(),
        },
        'training_task': 'final_state',
    }

    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as ckpt:
        torch.save(checkpoint_payload, ckpt.name)
        ckpt_path = ckpt.name

    try:
        predictor = ConditionDamagePredictor(
            structure_file=structure_file,
            dataset_npz=None,
            checkpoint_path=ckpt_path,
            model_type='gt',
            feature_mode='response',
            device='cpu',
        )

        assert predictor.seq_len == seq_len
        assert np.allclose(predictor.disp_mean, 0.0)
        assert np.allclose(predictor.disp_std, 1.0)
    finally:
        os.unlink(ckpt_path)
