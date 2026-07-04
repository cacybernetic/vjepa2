# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# V-JEPA 2.1 dense prediction objective:
#   L_dense = L_predict + lambda * L_ctx
# where L_predict is the original masked-token L1 loss and L_ctx is a
# distance-weighted L1 loss over the context tokens (Eq. 1-3 of the paper).

import torch

__all__ = [
    "separate_positions",
    "compute_mask_distance",
    "jepa_loss",
    "Lambda_LinearWarmupHold",
]


def _get_frame_pos(ids, grid_size):
    tokens_per_frame = int(grid_size * grid_size)
    return ids // tokens_per_frame


def _get_height_pos(ids, grid_size):
    tokens_per_frame = int(grid_size * grid_size)
    tokens_per_row = grid_size
    frame_ids = _get_frame_pos(ids, grid_size)
    ids = ids - tokens_per_frame * frame_ids
    return ids // tokens_per_row


def separate_positions(ids, grid_size):
    """Map flat token indices to ``(depth, height, width)`` grid coordinates."""
    tokens_per_frame = int(grid_size * grid_size)
    tokens_per_row = grid_size
    frame_ids = _get_frame_pos(ids, grid_size)
    height_ids = _get_height_pos(ids, grid_size)
    width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
    return 1.0 * frame_ids, 1.0 * height_ids, 1.0 * width_ids


def compute_mask_distance(masks_pred, masks_enc, grid_size, offset_context_loss=False):
    """Per-context-token weight ``sqrt(d_min)`` used in the context loss.

    For every context (unmasked) token we compute the minimum L2 distance, in
    grid blocks, to any masked/predicted token. The returned value is
    ``sqrt(d_min)`` so that dividing the loss by it yields ``lambda_i`` from
    Eq. 3 (``1/sqrt(d_min)``), emphasising context patches near masked regions.
    """
    distances = []
    for masks_pred_i, masks_enc_i in zip(masks_pred, masks_enc):
        row_distances = []
        for masks_pred_ij, masks_enc_ij in zip(masks_pred_i, masks_enc_i):
            N_enc_tokens = masks_enc_ij.shape[1]
            d_enc, h_enc, w_enc = separate_positions(masks_enc_ij, grid_size)
            d_pred, h_pred, w_pred = separate_positions(masks_pred_ij, grid_size)
            pred = torch.stack([d_pred, h_pred, w_pred], dim=-1)  # (B, N_pred, 3)
            enc_distances = []
            for enc_token in range(N_enc_tokens):
                enc_position = torch.stack(
                    [d_enc[:, enc_token], h_enc[:, enc_token], w_enc[:, enc_token]],
                    dim=-1,
                ).unsqueeze(1)  # (B, 1, 3)
                dist = torch.cdist(enc_position, pred, p=2)
                dmin, _ = dist.min(dim=-1)
                if offset_context_loss:
                    # Guard against grids smaller than the reference 16x16
                    # (grid_size // 16 == 0 would divide by zero).
                    coeff = max(grid_size // 16, 1)
                    dmin = dmin * (1.0 / coeff)
                dmin = dmin**0.5
                enc_distances.append(dmin)
            enc_distances = torch.stack(enc_distances, dim=-1).squeeze(1)
            row_distances.append(enc_distances)
        distances.append(row_distances)
    return distances


def jepa_loss(z, h, masks_to_apply, loss_exp=1.0, d_weights=None):
    """L1(-ish) loss between predictions ``z`` and (masked) targets ``h``.

    :param z: nested list ``[fpc][mask]`` of prediction tensors ``(B, K, D)``.
    :param h: list ``[fpc]`` of target tensors ``(B, N, D)`` (full sequence).
    :param masks_to_apply: nested list ``[fpc][mask]`` of index tensors selecting
        the tokens of ``h`` to compare against, matching ``z``.
    :param loss_exp: exponent applied to the absolute error (1.0 = L1).
    :param d_weights: optional nested list of per-token weights ``sqrt(d_min)``;
        the loss is divided by this value to realise ``1/sqrt(d_min)`` weighting.
    """
    from vjepa2.modules.tensors import apply_masks

    h = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, masks_to_apply)]

    loss, n = 0.0, 0
    if d_weights is not None:
        for zi, hi, d_i in zip(z, h, d_weights):
            for zij, hij, d_ij in zip(zi, hi, d_i):
                loss_n = torch.abs(zij - hij) ** loss_exp * (1 / d_ij.unsqueeze(2))
                loss += torch.mean(loss_n) / loss_exp
                n += 1
    else:
        for zi, hi in zip(z, h):
            for zij, hij in zip(zi, hi):
                loss += torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
                n += 1
    loss /= max(n, 1)
    return loss


class Lambda_LinearWarmupHold:
    """Linear warmup of the context-loss coefficient, then hold constant.

    0 before ``start_iter``; linear ramp to ``lambda_value`` over
    ``[start_iter, end_iter]``; constant afterwards.
    """

    def __init__(
        self, lambda_value: float, start_iter: int = 15_000, end_iter: int = 30_000
    ):
        assert end_iter > start_iter, "end_iter must be > start_iter"
        self.lambda_value = float(lambda_value)
        self.start = int(start_iter)
        self.end = int(end_iter)
        self.span = self.end - self.start

    def value(self, global_iter: int) -> float:
        if global_iter < self.start:
            return 0.0
        if global_iter >= self.end:
            return self.lambda_value
        alpha = (global_iter - self.start) / self.span
        return self.lambda_value * alpha
