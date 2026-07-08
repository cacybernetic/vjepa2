# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reusable ONNX export helpers. The real, well-tested export logic lives in the
# ``entrypoints.exportencoder`` command; this module gives that logic clean,
# importable names so other code can reuse it without going through the CLI.
# Keeping one source of truth avoids two export paths that could drift apart.

from __future__ import annotations

from vjepa2.entrypoints import exportencoder as _impl

__all__ = [
    "EncoderForExport",
    "build_wrapper",
    "dummy_input",
    "export_encoder",
    "validate_geometry",
    "export_main",
]

# Clean, public aliases over the tested export implementation.
EncoderForExport = _impl._EncoderForExport
build_wrapper = _impl._build_wrapper
dummy_input = _impl._dummy_input
export_encoder = _impl._export
validate_geometry = _impl.validate_geometry
export_main = _impl.main
