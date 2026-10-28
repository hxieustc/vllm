# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import dataclasses
import sys
import types

import pytest

from ._paths import KVCR_SOURCE, PREFIX, direct_module


def test_spec_is_pinned_interface_zero_cpu_and_lazy_worker(
    api, config, monkeypatch, tmp_path
):
    monkeypatch.setenv("KVCR_DIRECT_META_DIR", str(tmp_path.parent))
    KVCRDirectGuardSpec = direct_module("spec").KVCRDirectGuardSpec

    spec = KVCRDirectGuardSpec(config)
    assert isinstance(spec, api.base.OffloadingSpec)
    assert f"{PREFIX}.worker" not in sys.modules
    calls = []

    class Worker:
        def __init__(self, cfg, caches):
            calls.append((cfg, caches))

    module = types.ModuleType(f"{PREFIX}.worker")
    monkeypatch.setattr(module, "DirectGuardWorker", Worker, raising=False)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    caches = object()
    worker = spec.get_worker(caches)
    assert spec.get_worker(caches) is worker and calls == [(config, caches)]
    m = spec.get_manager()
    assert spec.get_manager() is m
    m.shutdown()
    assert not hasattr(spec, "primary_kv_view")


@pytest.mark.parametrize("change", ["tp", "bpc", "cpu", "guard"])
def test_spec_rejects_unsupported_topology_and_capacity(api, config, change):
    KVCRDirectGuardSpec = direct_module("spec").KVCRDirectGuardSpec

    if change == "tp":
        config = dataclasses.replace(
            config, parallel=dataclasses.replace(config.parallel, tp_size=2)
        )
    if change == "bpc":
        config = dataclasses.replace(
            config, cache=dataclasses.replace(config.cache, blocks_per_chunk=2)
        )
    if change == "cpu":
        config = dataclasses.replace(
            config, extra_config={**config.extra_config, "cpu_bytes_to_use": 1}
        )
    if change == "guard":
        config = dataclasses.replace(
            config,
            extra_config={**config.extra_config, "guard_pool_bytes": 125 * 1024**3},
        )
    with pytest.raises(ValueError):
        KVCRDirectGuardSpec(config)


def test_metric_definitions_reuse_old_helper_and_direct_counters(api):
    KVCRDirectGuardSpec = direct_module("spec").KVCRDirectGuardSpec

    definitions = KVCRDirectGuardSpec.build_metric_definitions(
        {"enable_telemetry": True}
    )
    for name, metadata in api.helper._kvcr_metric_definitions().items():
        assert definitions[name] == metadata
    assert "vllm:kvcr_direct_metadata_queries" in definitions
    assert all("request_id" not in d.labelnames for d in definitions.values())
    assert (
        KVCRDirectGuardSpec.build_metric_definitions({"enable_telemetry": False}) == {}
    )


def test_metric_definitions_do_not_import_obsolete_native_adapter(api, monkeypatch):
    import builtins

    KVCRDirectGuardSpec = direct_module("spec").KVCRDirectGuardSpec

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if (
            name == "kvcr"
            or name.startswith("kvcr.")
            or name.endswith("tiering.kvcr.manager")
        ):
            raise ImportError("legacy adapter imports removed MemDescriptor")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert "vllm:kvcr_state" in KVCRDirectGuardSpec.build_metric_definitions(
        {"enable_telemetry": True}
    )


def test_spec_accepts_secondary_tier_routing_metadata_without_constructing_tiers(
    api, config
):
    KVCRDirectGuardSpec = direct_module("spec").KVCRDirectGuardSpec

    routing = [
        {
            "type": "kvcr",
            "router_capabilities": ["router_hint"],
            "control_ports": list(range(17771, 17779)),
        }
    ]
    cfg = dataclasses.replace(
        config, extra_config={**config.extra_config, "secondary_tiers": routing}
    )
    spec = KVCRDirectGuardSpec(cfg)
    assert spec.config.extra_config["secondary_tiers"] == routing
    assert spec._manager is None and spec._worker is None


def test_spec_rejects_uncertified_canonical_layout(api, config):
    KVCRDirectGuardSpec = direct_module("spec").KVCRDirectGuardSpec

    with pytest.raises(ValueError, match="canonical"):
        KVCRDirectGuardSpec(dataclasses.replace(config, canonical_layout=True))


@pytest.mark.parametrize("key", ["guard_socket_path", "compatibility_digest"])
def test_spec_requires_explicit_guard_identity(api, config, key):
    KVCRDirectGuardSpec = direct_module("spec").KVCRDirectGuardSpec

    extra = dict(config.extra_config)
    extra.pop(key)
    with pytest.raises(ValueError, match=key):
        KVCRDirectGuardSpec(dataclasses.replace(config, extra_config=extra))


def test_actual_custom_core_emitted_metric_names_have_definitions(api):
    KVCRDirectGuardSpec = direct_module("spec").KVCRDirectGuardSpec

    if not (KVCR_SOURCE / "src/kvcr").is_dir():
        pytest.skip("Set KVCR_SOURCE_DIR to the harryx/guard-direct KVCR checkout")
    emitted = set()
    for path in (KVCR_SOURCE / "src/kvcr").rglob("*.py"):
        tree = ast.parse(path.read_text())
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = node.value.value
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Attribute)
                or not node.args
            ):
                continue
            if node.func.attr not in {
                "increase_counter",
                "set_gauge",
                "observe_histogram",
                "_record_progress_counter",
            }:
                continue
            arg = node.args[0]
            name = (
                arg.value
                if isinstance(arg, ast.Constant)
                else constants.get(arg.id)
                if isinstance(arg, ast.Name)
                else None
            )
            if isinstance(name, str) and name.startswith("kvcr_"):
                emitted.add(name)
    assert "kvcr_guard_source_events" in emitted
    definitions = KVCRDirectGuardSpec.build_metric_definitions(
        {"enable_telemetry": True}
    )
    assert {"vllm:" + name for name in emitted} <= definitions.keys()
    guard = definitions["vllm:kvcr_guard_source_events"]
    assert isinstance(guard, api.base.OffloadingCounterMetadata)
    assert guard.labelnames == ("event",)
