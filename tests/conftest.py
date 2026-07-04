import os

import pytest
import torch

WEIGHTS = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "weights",
    "vjepa2_1_vitb_dist_vitG_384.pt",
)


@pytest.fixture(scope="session")
def weights_path():
    if not os.path.exists(WEIGHTS):
        pytest.skip(f"checkpoint not found: {WEIGHTS}")
    return WEIGHTS


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
