# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Self-contained port of the tensor / masking helpers used by V-JEPA 2.1.
# (Originally spread across ``src/utils/tensors.py`` and ``src/masks/utils.py``
# in the reference implementation.)

import math

import torch

__all__ = [
    "trunc_normal_",
    "repeat_interleave_batch",
    "apply_masks",
]


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on
    # https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [lower, upper], then translate
        # to [2*lower-1, 2*upper-1].
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal.
        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """In-place truncated-normal initialization."""
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def repeat_interleave_batch(x, B, repeat):
    """Repeat-interleave a batched tensor keeping batch groups contiguous."""
    N = len(x) // B
    x = torch.cat(
        [
            torch.cat([x[i * B : (i + 1) * B] for _ in range(repeat)], dim=0)
            for i in range(N)
        ],
        dim=0,
    )
    return x


def apply_masks(x, masks, concat=True):
    """
    Keep only the patch tokens indexed by ``masks``.

    :param x: tensor of shape ``[B, N, D]``
    :param masks: list of tensors of shape ``[B, K]`` with indices of the K
        patches (out of N) to keep.
    :param concat: if True concatenate the kept-token tensors along the batch
        dimension, otherwise return them as a list.
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x
    return torch.cat(all_x, dim=0)
