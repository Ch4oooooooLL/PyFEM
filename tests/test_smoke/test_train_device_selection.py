from __future__ import annotations

import torch

from Deep_learning.train import _resolve_cuda_device_index


def test_resolve_cuda_device_index_defaults_plain_cuda_to_zero() -> None:
    assert _resolve_cuda_device_index(torch.device("cuda")) == 0


def test_resolve_cuda_device_index_keeps_explicit_cuda_index() -> None:
    assert _resolve_cuda_device_index(torch.device("cuda:1")) == 1
