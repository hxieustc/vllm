# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute pinned framework helper bodies with explicit CPU-only data/import fakes."""

import ast
import copy
import importlib.util
import json
import math
import sys
from collections import defaultdict
from enum import Enum
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as N

import pytest

from ._paths import ADAPTER, PREFIX, ROOT

INSTALLED = ROOT / "vllm"
DYNAMO = Path(__file__).parent / "fixtures/dynamo_cache_info.py"


def execute(path, names, namespace):
    tree = ast.parse(path.read_text())
    tree.body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ] + [
        node
        for node in tree.body
        if getattr(node, "name", None) in names
        or isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
    ]
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def runtime(monkeypatch):
    class Spec:
        def __init__(self, block_size):
            self.block_size = block_size
            self.prefix_cacheable = True
            self.tokens_per_state = 1
            self.dcp_sharded = False
            self.sliding_window = None

    classes = {"KVCacheSpec": Spec}
    for name, parent in [
        ("FullAttentionSpec", "KVCacheSpec"),
        ("MLAAttentionSpec", "FullAttentionSpec"),
        ("HiddenStateCacheSpec", "MLAAttentionSpec"),
        ("SinkFullAttentionSpec", "FullAttentionSpec"),
        ("SlidingWindowSpec", "KVCacheSpec"),
        ("SlidingWindowMLASpec", "SlidingWindowSpec"),
        ("MambaSpec", "KVCacheSpec"),
        ("ChunkedLocalAttentionSpec", "KVCacheSpec"),
        ("EncoderOnlyAttentionSpec", "KVCacheSpec"),
        ("CrossAttentionSpec", "KVCacheSpec"),
        ("UniformTypeKVCacheSpecs", "KVCacheSpec"),
    ]:
        classes[name] = type(name, (classes[parent],), {})
    kinds = execute(
        INSTALLED / "v1/kv_cache_interface.py",
        {"KVCacheSpecKind", "get_kv_cache_spec_kind"},
        dict(classes, Enum=Enum),
    )
    resolve = execute(
        ROOT / "vllm/v1/core/kv_cache_utils.py",
        {"resolve_dcp_kv_block_size", "resolve_kv_cache_block_sizes"},
        dict(
            classes,
            math=math,
            logger=N(info=lambda *args: None),
            iter_layer_specs=lambda spec: (spec,),
        ),
    )
    cache = execute(
        DYNAMO,
        {
            "MAIN_ATTENTION_KV_CACHE_KINDS",
            "DYNAMO_KV_EVENT_BLOCK_SIZE_KEY",
            "get_configured_kv_event_block_size",
            "select_main_attention_block_size",
        },
        {},
    )
    for name, namespace in [
        ("vllm.v1.kv_cache_interface", kinds),
        ("vllm.v1.core.kv_cache_utils", resolve),
        ("dynamo.vllm.cache_info", cache),
    ]:
        module = ModuleType(name)
        module.__dict__.update(namespace)
        monkeypatch.setitem(sys.modules, name, module)
    path = ADAPTER / "router_geometry.py"
    spec = importlib.util.spec_from_file_location("test_router_geometry_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pc = N(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        world_size=1,
        data_parallel_rank_local=2,
        data_parallel_index=2,
    )
    cfg = N(
        parallel_config=pc,
        cache_config=N(
            block_size=256, prefix_match_unit=None, enable_prefix_caching=True
        ),
        additional_config={},
        kv_transfer_config=N(
            kv_connector_extra_config={
                "router_hint_block_size": 256,
                "blocks_per_chunk": 1,
            }
        ),
    )
    specs = [
        classes["SlidingWindowMLASpec"](64),
        classes["SlidingWindowMLASpec"](64),
        classes["MLAAttentionSpec"](256),
        classes["SlidingWindowMLASpec"](8),
        classes["HiddenStateCacheSpec"](4),
    ]
    kv = N(kv_cache_groups=[N(kv_cache_spec=spec) for spec in specs])
    return N(
        module=module,
        cfg=cfg,
        kv=kv,
        classes=classes,
        kinds=kinds,
        resolve=resolve,
        cache=cache,
    )


def test_actual_kind_selection_and_hash_helpers_agree_for_certified_geometry(runtime):
    r = runtime
    before = (
        copy.deepcopy(vars(r.cfg.cache_config)),
        copy.deepcopy(r.cfg.additional_config),
    )
    proof = r.module.validate_router_geometry(r.cfg, r.kv, require_fallback=True)
    assert proof["verified"] and proof["frontend_fallback_verified"]
    assert [g["block_size"] for g in proof["group_metadata"]] == [64, 64, 256, 8, 4]
    assert proof["group_metadata"][0]["kind"] == "sliding_window_mla"
    assert (
        proof["metadata_selected_block_size"] == 256
        and proof["fallback_block_size_here"] == 256
    )
    assert proof["configured_block_size_here"] == 256
    assert proof["tokens_per_hash"] == 4 and proof["scheduler_block_size"] == 256
    assert len(proof["sources"]) >= 3
    assert before == (vars(r.cfg.cache_config), r.cfg.additional_config)


def test_projection_matches_pinned_manager_init_and_engine_metadata(runtime):
    r = runtime

    def method(path, cls, name, namespace):
        tree = ast.parse(path.read_text())
        owner = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls
        )
        node = next(
            n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        node.decorator_list = []
        body = [
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ]
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                str(path),
                "exec",
            ),
            namespace,
        )
        return namespace[name]

    initialize = method(
        INSTALLED / "v1/core/single_type_kv_cache_manager.py",
        "SingleTypeKVCacheManager",
        "__init__",
        dict(defaultdict=defaultdict),
    )
    managers = []
    for i, g in enumerate(r.kv.kv_cache_groups):
        # Storage/state granularity must not replace the logical block span.
        g.kv_cache_spec.tokens_per_state = 4
        g.kv_cache_spec.storage_block_size = 1
        manager = N()
        initialize(manager, g.kv_cache_spec, N(null_block=None), True, i, 256, 1, 1)
        assert (
            manager.block_size
            == g.kv_cache_spec.block_size
            == r.resolve["resolve_dcp_kv_block_size"](g.kv_cache_spec, 1)
        )
        managers.append(manager)
    metadata = method(
        ROOT / "vllm/v1/engine/core.py",
        "EngineCore",
        "get_kv_cache_group_metadata",
        dict(
            cast=lambda _, x: x,
            Any=object,
            get_kv_cache_spec_kind=r.kinds["get_kv_cache_spec_kind"],
        ),
    )
    engine = N(
        scheduler=N(
            kv_cache_config=r.kv,
            kv_cache_manager=N(coordinator=N(single_type_managers=managers)),
        )
    )
    proof = r.module.validate_router_geometry(r.cfg, r.kv, require_fallback=True)
    assert metadata(engine) == proof["group_metadata"]
    assert proof["manager_metadata_observed"] is False


def test_captured_manager_subclasses_do_not_override_block_span():
    tree = ast.parse(
        (INSTALLED / "v1/core/single_type_kv_cache_manager.py").read_text()
    )
    owners = []
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        for node in ast.walk(cls):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Store)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr == "block_size"
            ):
                owners.append(cls.name)
    assert owners == ["SingleTypeKVCacheManager"]


def test_installed_api_smoke_runs_both_cases_with_explicit_import_fakes(
    runtime, monkeypatch
):
    r = runtime
    interface = sys.modules["vllm.v1.kv_cache_interface"]
    monkeypatch.setattr(
        interface,
        "FullAttentionSpec",
        lambda **kw: r.classes["FullAttentionSpec"](kw["block_size"]),
        raising=False,
    )
    monkeypatch.setattr(interface, "KVCacheConfig", lambda **kw: N(**kw), raising=False)
    monkeypatch.setattr(
        interface, "KVCacheGroupSpec", lambda **kw: N(**kw), raising=False
    )

    def wrapper(**kw):
        obj = r.classes["UniformTypeKVCacheSpecs"](kw["block_size"])
        obj.kv_cache_specs = kw["kv_cache_specs"]
        return obj

    monkeypatch.setattr(interface, "UniformTypeKVCacheSpecs", wrapper, raising=False)
    monkeypatch.setitem(sys.modules, "torch", N(float16="explicit CPU dtype fake"))
    monkeypatch.setitem(
        sys.modules,
        f"{PREFIX}.router_geometry",
        r.module,
    )
    path = (
        ROOT / "tests/v1/kv_connector/kvcr_integration/direct_router_geometry_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("geometry_smoke", path)
    assert spec is not None and spec.loader is not None
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    result = smoke.run_smoke()
    assert result["status"] == "passed"
    assert result["positive"]["tokens_per_hash"] == 256
    assert result["first_main_64_rejected"]
    assert (
        result["worker_uniform_positive"]["group_metadata"]
        == result["positive"]["group_metadata"]
    )
    assert result["worker_uniform_first_main_64_rejected"]


def test_actual_kind_helper_handles_worker_uniform_representation(runtime):
    r = runtime
    original = r.module.validate_router_geometry(r.cfg, r.kv, require_fallback=True)
    for group in r.kv.kv_cache_groups:
        member = group.kv_cache_spec
        wrapper = r.classes["UniformTypeKVCacheSpecs"](member.block_size)
        wrapper.kv_cache_specs = {"layer0": member, "layer1": copy.deepcopy(member)}
        group.kv_cache_spec = wrapper
    wrapped = r.module.validate_router_geometry(r.cfg, r.kv, require_fallback=True)
    assert original["group_metadata"] == wrapped["group_metadata"]
    assert wrapped["tokens_per_hash"] == 4


@pytest.mark.parametrize(
    "case",
    [
        "first_main_64",
        "no_main",
        "override",
        "fallback",
        "cp",
        "tp",
        "hash_override",
        "hint_size",
    ],
)
def test_wrong_main_kind_or_frontend_override_fails_closed(runtime, case):
    r = runtime
    if case == "first_main_64":
        r.kv.kv_cache_groups[0].kv_cache_spec = r.classes["FullAttentionSpec"](64)
    elif case == "no_main":
        for group in r.kv.kv_cache_groups:
            group.kv_cache_spec = r.classes["SlidingWindowMLASpec"](
                group.kv_cache_spec.block_size
            )
    elif case == "override":
        r.cfg.additional_config["dynamo_kv_event_block_size"] = 64
    elif case == "fallback":
        r.cfg.cache_config.block_size = 64
    elif case == "cp":
        r.cfg.parallel_config.decode_context_parallel_size = 2
    elif case == "tp":
        r.cfg.parallel_config.tensor_parallel_size = 2
    elif case == "hash_override":
        r.cfg.cache_config.prefix_match_unit = 3
    elif case == "hint_size":
        r.cfg.kv_transfer_config.kv_connector_extra_config["router_hint_block_size"] = (
            64
        )
    with pytest.raises(ValueError):
        r.module.validate_router_geometry(r.cfg, r.kv, require_fallback=True)


def test_scheduler_minimum_is_not_claimed_as_frontend_fallback(runtime):
    r = runtime
    r.cfg.cache_config.block_size = 4
    proof = r.module.validate_router_geometry(r.cfg, r.kv, require_fallback=False)
    assert proof["metadata_selected_block_size"] == 256
    assert (
        proof["fallback_block_size_here"] == 4
        and proof["configured_block_size_here"] == 4
    )
    assert proof["frontend_fallback_verified"] is False
    r.cfg.additional_config["dynamo_kv_event_block_size"] = 64
    with pytest.raises(ValueError):
        r.module.validate_router_geometry(r.cfg, r.kv, require_fallback=False)


def test_connector_validates_before_base_factory_and_writes_actual_identity(
    runtime, tmp_path, monkeypatch
):
    r = runtime
    calls = []

    def attest(*args, **kwargs):
        calls.append("geometry")
        return r.module.validate_router_geometry(*args, **kwargs)

    class Base:
        def __init__(self, *args):
            calls.append("base")
            self.connector_worker = N(spec="spec")

    class Worker:
        def __init__(self, *args):
            calls.append("worker")

    tree = ast.parse((ADAPTER / "connector.py").read_text())
    tree.body = [
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "KVCRDirectGuardConnector"
    ]
    namespace = dict(
        OffloadingConnector=Base,
        DirectConnectorWorker=Worker,
        KVConnectorRole=N(WORKER="worker"),
        attest_router_geometry=attest,
        write_router_geometry_proof=r.module.write_router_geometry_proof,
    )
    exec(compile(tree, "real_connector_class", "exec"), namespace)
    monkeypatch.setenv("RUN_DIR", str(tmp_path))
    namespace["KVCRDirectGuardConnector"](r.cfg, "worker", r.kv)
    assert calls == ["geometry", "base", "worker"]
    files = list(tmp_path.glob("router-geometry-*-2.json"))
    assert len(files) == 1
    proof = json.loads(files[0].read_text())
    assert proof["rank"] == 2 and proof["pid"] > 0
    calls.clear()
    r.kv.kv_cache_groups[0].kv_cache_spec = r.classes["FullAttentionSpec"](64)
    with pytest.raises(ValueError):
        namespace["KVCRDirectGuardConnector"](r.cfg, "worker", r.kv)
    assert calls == ["geometry"]
