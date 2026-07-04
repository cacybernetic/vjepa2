"""Tests for the V-JEPA 2.1 dense prediction objective."""

import torch

from vjepa2.modules.losses import (
    Lambda_LinearWarmupHold,
    compute_mask_distance,
    jepa_loss,
    separate_positions,
)


def test_separate_positions_grid():
    # grid_size 4 -> tokens_per_frame 16, tokens_per_row 4
    ids = torch.tensor([[0, 5, 17]])
    d, h, w = separate_positions(ids, grid_size=4)
    # token 0 -> (0,0,0); token 5 -> (0,1,1); token 17 -> (1,0,1)
    assert torch.equal(d, torch.tensor([[0.0, 0.0, 1.0]]))
    assert torch.equal(h, torch.tensor([[0.0, 1.0, 0.0]]))
    assert torch.equal(w, torch.tensor([[0.0, 1.0, 1.0]]))


def test_compute_mask_distance_values():
    grid = 4
    masks_enc = [[torch.tensor([[0, 1]])]]   # context tokens (0,0,0),(0,0,1)
    masks_pred = [[torch.tensor([[2]])]]     # masked token   (0,0,2)
    d = compute_mask_distance(masks_pred, masks_enc, grid_size=grid)
    dmin = d[0][0]  # sqrt of min distance in blocks
    # distances are 2 and 1 -> sqrt = sqrt(2), 1
    assert torch.allclose(dmin, torch.tensor([[2.0**0.5, 1.0]]), atol=1e-5)


def test_jepa_loss_zero_on_perfect_prediction():
    N = 8
    h = [torch.randn(1, N, 4)]
    mask = torch.arange(N).unsqueeze(0)
    z = [[h[0].clone()]]
    loss = jepa_loss(z, h, [[mask]])
    assert loss.item() == 0.0


def test_jepa_loss_is_l1():
    h = [torch.zeros(1, 3, 2)]
    z = [[torch.ones(1, 3, 2)]]
    loss = jepa_loss(z, h, [[torch.arange(3).unsqueeze(0)]])
    assert abs(loss.item() - 1.0) < 1e-6


def test_jepa_loss_distance_weighting_reduces_far_tokens():
    # Far context tokens (large d) should contribute less to the loss.
    h = [torch.zeros(1, 2, 1)]
    z = [[torch.ones(1, 2, 1)]]
    mask = torch.arange(2).unsqueeze(0)
    near = [[torch.tensor([[1.0, 1.0]])]]   # d=1 for both
    far = [[torch.tensor([[1.0, 4.0]])]]    # second token far (weight 1/4)
    loss_near = jepa_loss(z, h, [[mask]], d_weights=near)
    loss_far = jepa_loss(z, h, [[mask]], d_weights=far)
    assert loss_far.item() < loss_near.item()


def test_lambda_warmup_schedule():
    sched = Lambda_LinearWarmupHold(lambda_value=0.5, start_iter=10, end_iter=20)
    assert sched.value(5) == 0.0
    assert abs(sched.value(15) - 0.25) < 1e-6
    assert sched.value(25) == 0.5
