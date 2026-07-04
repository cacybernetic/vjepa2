# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["VJEPA21", "init_video_model", "build_vjepa2_1_vitb"]


def __getattr__(name):
    # Import the (torch-heavy) model lazily so torch-free entrypoints — e.g. the
    # ONNX-only ``runs`` inference CLI — can import the package without pulling
    # in torch. See PEP 562.
    if name in __all__:
        from vjepa2 import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
