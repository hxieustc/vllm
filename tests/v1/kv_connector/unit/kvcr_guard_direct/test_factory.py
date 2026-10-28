# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise production factory registration without a CUDA framework import."""

import ast
import importlib
from types import SimpleNamespace

from ._paths import PREFIX, ROOT, direct_module


def factory_from_source(path, class_name, registration, namespace):
    tree = ast.parse(path.read_text())
    tree.body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        *[
            node
            for node in tree.body
            if (isinstance(node, ast.ClassDef) and node.name == class_name)
            or (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == registration
            )
        ],
    ]
    namespace.update(importlib=importlib)
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace[class_name]


def test_registered_spec_loads_in_tree_without_native_or_dynamo_imports(
    api, monkeypatch
):
    import builtins

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"kvcr", "nixl", "dynamo"}:
            raise AssertionError(f"Eager optional dependency import: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    logger = SimpleNamespace(info=lambda *args: None, warning_once=lambda *args: None)
    factory = factory_from_source(
        ROOT / "vllm/v1/kv_offload/factory.py",
        "OffloadingSpecFactory",
        "register_spec",
        dict(OffloadingSpec=api.base.OffloadingSpec, logger=logger),
    )
    cls = factory.get_spec_cls({"spec_name": "KVCRDirectGuardSpec"})
    assert cls is direct_module("spec").KVCRDirectGuardSpec
    assert issubclass(cls, api.base.OffloadingSpec)


def test_registered_connector_resolves_without_external_module_path(api, monkeypatch):
    factory = factory_from_source(
        ROOT / "vllm/distributed/kv_transfer/kv_connector/factory.py",
        "KVConnectorFactory",
        "register_connector",
        {},
    )
    imports = []
    expected = type("KVCRDirectGuardConnector", (), {})

    def load(name):
        imports.append(name)
        return SimpleNamespace(KVCRDirectGuardConnector=expected)

    monkeypatch.setattr(importlib, "import_module", load)
    config = SimpleNamespace(
        kv_connector="KVCRDirectGuardConnector", kv_connector_module_path=None
    )
    assert factory.get_connector_class(config) is expected
    assert imports == [f"{PREFIX}.connector"]
