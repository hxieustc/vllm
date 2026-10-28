# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve in-tree adapter and optional paired KVCR sources for CPU tests."""

import importlib
import os
from pathlib import Path

ROOT = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "vllm/v1/kv_offload/base.py").is_file()
)
ADAPTER = ROOT / "vllm/distributed/kv_transfer/kv_connector/v1/kvcr/guard_direct"
KVCR_SOURCE = Path(os.environ.get("KVCR_SOURCE_DIR", ROOT.parent / "kvcr"))

PREFIX = "vllm.distributed.kv_transfer.kv_connector.v1.kvcr.guard_direct"


def direct_module(name):
    """Import the production module through its registered in-tree namespace."""
    return importlib.import_module(f"{PREFIX}.{name}")
