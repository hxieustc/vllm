# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU interface tests: real pinned base/stats bodies, explicit CUDA/import fakes."""

import ast
import dataclasses
import importlib
import importlib.util
import sys
import types
from typing import Any

import msgspec
import pytest
import zmq

from ._paths import KVCR_SOURCE, PREFIX, ROOT, direct_module


@pytest.fixture
def api(monkeypatch):
    snapshot = ROOT / "vllm"
    prior_direct = {
        name: value
        for name, value in sys.modules.items()
        if name == PREFIX or name.startswith(PREFIX + ".")
    }
    prior_base = sys.modules.get("vllm.v1.kv_offload.base")

    def module(name, **attrs):
        obj = types.ModuleType(name)
        obj.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, obj)
        return obj

    for name in list(sys.modules):
        if name == PREFIX or name.startswith(PREFIX + "."):
            monkeypatch.delitem(sys.modules, name)
    for name in [
        "vllm",
        "vllm.v1",
        "vllm.v1.kv_offload",
        "vllm.distributed",
        "vllm.distributed.kv_transfer",
        "vllm.distributed.kv_transfer.kv_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1",
        "vllm.distributed.kv_transfer.kv_connector.v1.kvcr",
    ]:
        suffix = name.split(".")[1:]
        module(name, __path__=[str(snapshot.joinpath(*suffix))])
    if not (KVCR_SOURCE / "src/kvcr").is_dir():

        @dataclasses.dataclass
        class MemoryRef:
            end_point_name: str
            element_index: int
            label: str = ""

        module("kvcr", __path__=[])
        module("kvcr.types", MemoryRef=MemoryRef)
    module("torch", Tensor=object)
    module("vllm.v1.kv_hints", KvHintsEnvelope=object)
    source_path = snapshot / "v1/metrics/cache_hit_source.py"
    sources = module("vllm.v1.metrics.cache_hit_source")
    exec(compile(source_path.read_text(), str(source_path), "exec"), sources.__dict__)
    config = importlib.import_module("vllm.v1.kv_offload.config")
    # Base and adapter must share this test's actual enum class identity.
    monkeypatch.delitem(sys.modules, "vllm.v1.kv_offload.base", raising=False)
    base = importlib.import_module("vllm.v1.kv_offload.base")

    # Execute the pinned stats implementation, replacing only its framework base.
    @dataclasses.dataclass
    class KVConnectorStats:
        data: dict = dataclasses.field(default_factory=dict)

    stats_name = "vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics"
    source = snapshot / "distributed/kv_transfer/kv_connector/v1/offloading/metrics.py"
    tree = ast.parse(source.read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name in {"_MetricType", "_StatsKey", "OffloadingConnectorStats"}
    ]
    stats = module(
        stats_name,
        dataclass=dataclasses.dataclass,
        Any=Any,
        KVConnectorStats=KVConnectorStats,
    )
    exec(compile(tree, str(source), "exec"), stats.__dict__)
    # Reuse the exact old adapter metric helper without importing native KVCR.
    helper_name = "vllm.v1.kv_offload.tiering.kvcr.manager"
    helper_path = snapshot / "v1/kv_offload/tiering/kvcr/manager.py"
    helper_tree = ast.parse(helper_path.read_text())
    helper_tree.body = [
        n
        for n in helper_tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_vllm_metric_name", "_kvcr_metric_definitions"}
    ]
    helper = module(helper_name, **base.__dict__)
    helper.__dict__.update(
        DURATION_METRIC="kvcr_duration_seconds",
        TRANSFER_BYTES_METRIC="kvcr_transfer_bytes",
        TRANSFER_BLOCKS_METRIC="kvcr_transfer_blocks",
        STATE_METRIC="kvcr_state",
    )
    exec(compile(helper_tree, str(helper_path), "exec"), helper.__dict__)
    yield types.SimpleNamespace(
        base=base, config=config, stats=stats, helper=helper, sources=sources
    )
    # Do not leave modules bound to fake framework types for unrelated tests.
    for name in list(sys.modules):
        if name == PREFIX or name.startswith(PREFIX + "."):
            sys.modules.pop(name, None)
    sys.modules.update(prior_direct)
    if prior_base is None:
        sys.modules.pop("vllm.v1.kv_offload.base", None)


@pytest.fixture
def config(api):
    c = api.config
    return c.OffloadingConfig(
        groups=(c.OffloadingGroupConfig(256, ("layer",), 0),),
        worker_kv_bytes_per_block=4096,
        enable_kv_cache_events=True,
        extra_config={
            "cpu_bytes_to_use": 0,
            "guard_pool_bytes": 124 * 1024**3,
            "enable_telemetry": True,
            "guard_socket_path": "/run/kvcr/memory.sock",
            "compatibility_digest": "direct-test-v1",
        },
        engine_id="engine-a",
        model=c.OffloadingModelConfig("test", "fp8"),
        cache=c.OffloadingCacheConfig(256, 1),
        parallel=c.OffloadingParallelConfig(0, 1, 1, 1, 1, 1, 0, 8, None, True),
    )


def test_endpoint_is_stable_dp_scoped_and_requires_explicit_directory(
    api, config, monkeypatch, tmp_path
):
    metadata_endpoint = direct_module("protocol").metadata_endpoint

    monkeypatch.delenv("KVCR_DIRECT_META_DIR", raising=False)
    with pytest.raises(ValueError, match="KVCR_DIRECT_META_DIR"):
        metadata_endpoint(config)
    monkeypatch.setenv("KVCR_DIRECT_META_DIR", str(tmp_path.parent))
    first = metadata_endpoint(config)
    assert first.startswith("ipc://" + str(tmp_path.parent) + "/")
    assert metadata_endpoint(config) == first
    other = dataclasses.replace(
        config, parallel=dataclasses.replace(config.parallel, data_parallel_index=1)
    )
    assert metadata_endpoint(other) != first


def test_wire_preserves_bytes_and_two_frame_router_contract(
    api, config, monkeypatch, tmp_path
):
    MetadataClient = direct_module("protocol").MetadataClient
    Request = direct_module("protocol").Request
    Response = direct_module("protocol").Response
    metadata_endpoint = direct_module("protocol").metadata_endpoint

    monkeypatch.setenv("KVCR_DIRECT_META_DIR", str(tmp_path.parent))
    ctx = zmq.Context()
    server = ctx.socket(zmq.ROUTER)
    server.bind(metadata_endpoint(config))
    client = MetadataClient(config, context=ctx)
    try:
        rid = client.request("query", "request-a", (b"\x00\xff", b"b"))
        client.flush()
        assert server.poll(1000)
        frames = server.recv_multipart()
        assert len(frames) == 2
        ident, body = frames
        request = msgspec.msgpack.decode(body, type=Request)
        assert (request.id, request.kind, request.keys) == (
            rid,
            "query",
            [b"\x00\xff", b"b"],
        )
        reply = Response(rid, statuses=[["HIT", "LOCAL_G2"], ["FETCHING", None]])
        server.send_multipart([ident, msgspec.msgpack.encode(reply)])
        assert client.socket.poll(1000)
        assert client.drain() == [reply]
        assert not client.has_pending_work()
    finally:
        client.close()
        server.close(0)
        ctx.term()


def test_client_bounds_pending_requests_and_rejects_unknown_response(
    api, config, monkeypatch, tmp_path
):
    MetadataClient = direct_module("protocol").MetadataClient
    Response = direct_module("protocol").Response
    metadata_endpoint = direct_module("protocol").metadata_endpoint

    monkeypatch.setenv("KVCR_DIRECT_META_DIR", str(tmp_path.parent))
    ctx = zmq.Context()
    server = ctx.socket(zmq.ROUTER)
    server.bind(metadata_endpoint(config))
    client = MetadataClient(config, context=ctx, max_pending=1)
    try:
        client.request("poll")
        with pytest.raises(BufferError):
            client.request("poll")
        client.flush()
        assert server.poll(1000)
        ident, _ = server.recv_multipart()
        server.send_multipart([ident, msgspec.msgpack.encode(Response(999))])
        assert client.socket.poll(1000)
        with pytest.raises(ValueError, match="response"):
            client.drain()
    finally:
        client.close()
        client.close()
        server.close(0)
        ctx.term()


def test_direct_spec_is_pinned_loadstore_dataclass(api):
    DirectSpec = direct_module("protocol").DirectSpec

    value = DirectSpec((b"key",), "req")
    assert isinstance(value, api.base.LoadStoreSpec)
    assert value.keys == (b"key",) and value.request_id == "req"
    assert dataclasses.is_dataclass(value)
