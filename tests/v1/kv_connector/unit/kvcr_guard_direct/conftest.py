# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise production source with test-scoped CPU import interfaces."""

import importlib
import sys

from ._paths import KVCR_SOURCE

sys.path.insert(0, str(KVCR_SOURCE / "src"))
# Load native import dependencies before per-test Torch modules are replaced.
if (KVCR_SOURCE / "src/kvcr").is_dir():
    importlib.import_module("kvcr")
import pytest

from .test_protocol import api, config  # noqa: E402,F401


@pytest.fixture(autouse=True)
def _cpu_interfaces(request):
    """Keep package and CUDA fakes scoped to each test, not global collection."""
    request.getfixturevalue("api")
