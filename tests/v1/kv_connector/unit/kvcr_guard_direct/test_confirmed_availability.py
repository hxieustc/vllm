# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Confirmed Guard admission, explicit CPU fakes, exact pinned HMA bodies."""

import ast
import dataclasses
import time
from time import monotonic
from types import SimpleNamespace as N

import msgspec
import pytest
import zmq

from ._paths import KVCR_SOURCE, direct_module
from .test_hint_scope import external, geometry, key, pinned_hma, positions
from .test_manager import FakeClient, tick


class Clock:
    now = 10.0

    def __call__(self):
        return self.now


def test_staggered_misses_converge_for_one_request(api, config):
    """A slow remote probe must not erase an earlier compute decision."""
    DirectGuardManager = direct_module("manager").DirectGuardManager

    clock = Clock()
    peer = FakeClient(clock=clock)
    manager = DirectGuardManager(config, client=peer, clock=clock)
    req = api.base.ReqContext("staggered")
    assert manager.lookup(b"first", req) is api.base.LookupResult.RETRY
    assert manager.lookup(b"second", req) is api.base.LookupResult.RETRY
    tick(api, manager)
    peer.answer(
        next(r for r in peer.sent if r.kind == "query"),
        statuses=[["MISS", None], ["FETCHING", "REMOTE_G2"]],
    )
    tick(api, manager)
    assert manager.lookup(b"first", req) is api.base.LookupResult.MISS
    clock.now = 14.9
    assert manager.lookup(b"second", req) is api.base.LookupResult.RETRY
    tick(api, manager)
    clock.now = 15.2
    peer.answer(
        [r for r in peer.sent if r.kind == "query"][-1], statuses=[["MISS", None]]
    )
    tick(api, manager)
    assert manager.lookup(b"first", req) is api.base.LookupResult.MISS
    assert manager.lookup(b"second", req) is api.base.LookupResult.MISS


def actor_response(message, client):
    DirectIOActor = direct_module("worker").DirectIOActor

    actor = DirectIOActor(N(extra_config={}), N(page_bytes=16))
    actor.client = client
    client.get_stats = lambda: None
    if not hasattr(client, "query_ready_observations"):

        def observations(keys, request_id):
            now = monotonic()
            return [
                dict(
                    status="FETCHABLE",
                    observed_at_monotonic=now,
                    valid_until_monotonic=now + 5,
                    request_generation="fake-hint-generation",
                    source_endpoint="tcp://source:17771",
                    source_incarnation="fake-source",
                    source_progress_epoch=0,
                    residency_generation=1,
                )
                for k in keys
            ]

        client.query_ready_observations = observations

    class Socket:
        incoming = [[b"peer", msgspec.msgpack.encode(message)]]
        sent = []

        def recv_multipart(self, flags):
            if self.incoming:
                return self.incoming.pop()
            raise zmq.Again()

        def send_multipart(self, frames, flags):
            self.sent.append(frames)

    socket = Socket()
    actor._metadata(socket)
    return msgspec.msgpack.decode(socket.sent[0][1])


def test_actor_uses_only_query_ready_and_preserves_pending_remote(api):
    class Client:
        def query(self, *args):
            raise AssertionError("legacy query must not run")

        def query_ready(self, keys, request_id, timeout_ms=5000):
            assert (
                keys == [b"pending", b"ready", b"absent"]
                and request_id == "r"
                and timeout_ms == 5000
            )
            return [
                (N(name=s), N(name=t) if t else None)
                for s, t in [
                    ("FETCHING", "REMOTE_G2"),
                    ("FETCHABLE", "REMOTE_G2"),
                    ("MISS", None),
                ]
            ]

    response = actor_response(
        dict(
            id=1, kind="query", request_id="r", keys=[b"pending", b"ready", b"absent"]
        ),
        Client(),
    )
    assert "error" not in response
    assert response["statuses"] == [
        ["FETCHING", "REMOTE_G2"],
        ["FETCHABLE", "REMOTE_G2"],
        ["MISS", None],
    ]


def test_actor_missing_confirmed_api_fails_without_legacy_fallback(api):
    class Client:
        def query(self, *args):
            raise AssertionError("legacy query invoked")

    response = actor_response(
        dict(id=1, kind="query", request_id="r", keys=[b"k"]), Client()
    )
    assert "query_ready" in response["error"]


class ConfirmedPeer(FakeClient):
    def __init__(self, ready):
        super().__init__()
        self.ready = ready
        self.hints = {}
        self.pending_once = True

    def flush(self):
        peer = self

        class Client:
            def query(self, keys, request_id):
                # Reproduces r5: eligibility alone advertises every hinted key.
                return [(N(name="FETCHABLE"), N(name="REMOTE_G2")) for k in keys]

            def query_ready(self, keys, request_id, timeout_ms=5000):
                hashes = set(
                    peer.hints[request_id]["actions"][0]["payload"]["block_hashes"]
                )
                return [
                    (N(name="FETCHING"), N(name="REMOTE_G2"))
                    if peer.pending_once
                    else (N(name="FETCHABLE"), N(name="REMOTE_G2"))
                    if k in peer.ready and external(k) in hashes
                    else (N(name="MISS"), None)
                    for k in keys
                ]

        for request in self.sent:
            if request.id not in self.pending:
                continue
            if request.kind == "hint":
                self.hints[request.request_id] = request.hint
            response = (
                actor_response(
                    dict(
                        id=request.id,
                        kind=request.kind,
                        request_id=request.request_id,
                        keys=request.keys,
                    ),
                    Client(),
                )
                if request.kind == "query"
                else {}
            )
            assert "error" not in response
            self.answer(request, statuses=response.get("statuses", []))
        self.pending_once = False


def test_actual_hma_sparse_swa_confirmation_converges_to_available_parent(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    cfg, req, groups = geometry(api, config, parents=(256, 512), total=512)
    positions(req, groups)
    ready = {
        key(e, g)
        for g, ends in {
            0: (192, 256, 448, 512),
            1: (192, 256, 320, 448, 512),
            2: (256, 512),
            3: tuple(range(136, 257, 8)) + tuple(range(392, 513, 8)),
            4: (252, 256, 508, 512),
        }.items()
        for e in ends
    }
    peer = ConfirmedPeer(ready)
    manager = DirectGuardManager(cfg, client=peer)
    manager.on_new_request(req)
    hma = pinned_hma(api)()
    hma.manager = manager
    hma._lookup_groups = (2, 0, 1, 3, 4)
    hma._sliding_window_groups = (0, 1, 3, 4)
    hma._mamba_align_size = None
    hma._chunks_being_loaded = set()
    hma._events_tracker = N(record_lookup=lambda *args: None)
    windows = (2, 2, None, 16, 2)
    hma.config = N(
        kv_group_configs=[
            N(
                tokens_per_chunk=size,
                sliding_window_size_in_chunks=w,
                is_eagle_group=g == 1,
                load_window_size_in_chunks=lambda end, w=w: w,
            )
            for g, (size, w) in enumerate(zip((64, 64, 256, 8, 4), windows))
        ]
    )
    state = N(
        num_locally_computed_tokens=0,
        max_load_tokens=None,
        req=N(num_tokens=513, request_id=req.req_id),
        req_context=req,
        group_states=[N(offload_keys=ks) for ks in groups],
    )
    result = None
    for _ in range(8):
        result = hma._lookup_complete_chunks(state)
        if result is not None:
            break
        tick(api, manager)
    assert result == 256  # r5's eligibility-only lookup incorrectly selects 448.
    assert manager.lookup(key(384, 0), req) is api.base.LookupResult.MISS
    assert manager.lookup(key(256, 0), req) is api.base.LookupResult.HIT


@pytest.mark.parametrize(
    "status,tier",
    [
        ("HIT", "LOCAL_G2"),
        ("FETCHABLE", "REMOTE_G2"),
        ("FETCHABLE", "G3"),
        ("FETCHING", "REMOTE_G2"),
    ],
)
def test_all_metadata_classes_expire_without_retaining_key_capacity(
    api, config, status, tier
):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    clock = Clock()
    peer = FakeClient(clock=clock)
    cfg = dataclasses.replace(
        config, extra_config={**config.extra_config, "metadata_max_keys_per_request": 1}
    )
    manager = DirectGuardManager(cfg, client=peer, clock=clock)
    req = api.base.ReqContext("r")
    manager.lookup(b"old", req)
    tick(api, manager)
    query = next(r for r in peer.sent if r.kind == "query")
    peer.answer(query, statuses=[[status, tier]])
    tick(api, manager)
    clock.now += 5.001
    assert manager.lookup(b"new", req) is api.base.LookupResult.RETRY
    assert b"old" not in manager._requests["r"].statuses
    assert manager._requests["r"].known == {b"new"}


def test_confirmed_miss_holds_bounded_request_capacity_until_finish(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    clock = Clock()
    peer = FakeClient(clock=clock)
    cfg = dataclasses.replace(
        config, extra_config={**config.extra_config, "metadata_max_keys_per_request": 1}
    )
    manager = DirectGuardManager(cfg, client=peer, clock=clock)
    req = api.base.ReqContext("r")
    assert manager.lookup(b"old", req) is api.base.LookupResult.RETRY
    tick(api, manager)
    peer.answer(
        next(r for r in peer.sent if r.kind == "query"), statuses=[["MISS", None]]
    )
    tick(api, manager)
    clock.now += 5.001
    assert manager.lookup(b"old", req) is api.base.LookupResult.MISS
    with pytest.raises(BufferError, match="key bound"):
        manager.lookup(b"new", req)
    manager.on_request_finished(req)
    assert (
        manager.lookup(b"new", api.base.ReqContext("next"))
        is api.base.LookupResult.RETRY
    )


def test_delayed_reply_cannot_restart_observation_lifetime(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    clock = Clock()
    peer = FakeClient(clock=clock)
    manager = DirectGuardManager(config, client=peer, clock=clock)
    req = api.base.ReqContext("r")
    manager.lookup(b"k", req)
    tick(api, manager)
    query = next(r for r in peer.sent if r.kind == "query")
    clock.now += 5.001
    peer.answer(query, statuses=[["FETCHABLE", "REMOTE_G2"]])
    tick(api, manager)
    assert manager.lookup(b"k", req) is api.base.LookupResult.RETRY
    assert b"k" not in manager._requests["r"].statuses


def test_expired_observation_not_attributed_as_cache_hit(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    clock = Clock()
    peer = FakeClient(clock=clock)
    manager = DirectGuardManager(config, client=peer, clock=clock)
    req = api.base.ReqContext("r")
    manager.lookup(b"k", req)
    tick(api, manager)
    peer.answer(
        next(r for r in peer.sent if r.kind == "query"),
        statuses=[["FETCHABLE", "REMOTE_G2"]],
    )
    tick(api, manager)
    clock.now += 5.001
    assert (
        manager.get_load_source(b"k", req)
        is api.sources.CacheHitSource.EXTERNAL_UNSPECIFIED
    )


@pytest.mark.parametrize("ttl", [0, -1, 5001, True, "5000", float("inf")])
def test_metadata_lifetime_is_explicit_finite_and_bounded(api, config, ttl):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    cfg = dataclasses.replace(
        config, extra_config={**config.extra_config, "metadata_status_ttl_ms": ttl}
    )
    with pytest.raises(ValueError, match="metadata_status_ttl_ms"):
        DirectGuardManager(cfg, client=FakeClient())


def test_new_response_does_not_extend_older_cache_snapshot(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    clock = Clock()
    peer = FakeClient(clock=clock)
    manager = DirectGuardManager(config, client=peer, clock=clock)
    req = api.base.ReqContext("r")
    manager.lookup(b"old", req)
    tick(api, manager)
    old = next(r for r in peer.sent if r.kind == "query")
    clock.now += 3.0
    peer.answer(old, statuses=[["HIT", "LOCAL_G2"]])
    tick(api, manager)
    manager.lookup(b"new", req)
    tick(api, manager)
    new = [r for r in peer.sent if r.kind == "query"][-1]
    peer.answer(new, statuses=[["FETCHABLE", "REMOTE_G2"]])
    tick(api, manager)
    assert manager.lookup(b"old", req) is api.base.LookupResult.HIT
    clock.now += 2.001
    tick(api, manager)
    assert not manager._requests["r"].statuses and not manager._requests["r"].known
    assert manager.lookup(b"old", req) is api.base.LookupResult.RETRY


def test_expiration_never_releases_inflight_capacity_or_revives_quarantine(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    clock = Clock()
    peer = FakeClient(clock=clock)
    manager = DirectGuardManager(config, client=peer, clock=clock)
    req = api.base.ReqContext("r")
    manager.lookup(b"cached", req)
    tick(api, manager)
    first = next(r for r in peer.sent if r.kind == "query")
    peer.answer(first, statuses=[["HIT", "LOCAL_G2"]])
    tick(api, manager)
    manager.lookup(b"inflight", req)
    tick(api, manager)
    late = [r for r in peer.sent if r.kind == "query"][-1]
    clock.now += 5.001
    tick(api, manager)
    assert manager._requests["r"].known == {b"inflight"}
    manager.mark_load_failure("r")
    peer.answer(late, statuses=[["FETCHABLE", "REMOTE_G2"]])
    tick(api, manager)
    assert not manager._requests["r"].known
    assert manager.lookup(b"cached", req) is api.base.LookupResult.MISS


def test_core_absolute_probe_timeout_becomes_miss_not_admission(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    class DeadlinePeer(FakeClient):
        def flush(self):
            now = clock.now

            class Client:
                def query_ready(self, keys, request_id, timeout_ms=5000):
                    return [
                        (N(name="MISS"), None)
                        if now >= 15
                        else (N(name="FETCHING"), N(name="REMOTE_G2"))
                        for k in keys
                    ]

            for request in self.sent:
                if request.id not in self.pending:
                    continue
                response = (
                    actor_response(
                        dict(
                            id=request.id,
                            kind=request.kind,
                            request_id=request.request_id,
                            keys=request.keys,
                        ),
                        Client(),
                    )
                    if request.kind == "query"
                    else {}
                )
                self.answer(request, statuses=response.get("statuses", []))

    clock = Clock()
    peer = DeadlinePeer()
    manager = DirectGuardManager(config, client=peer, clock=clock)
    req = api.base.ReqContext("r")
    for step in range(6):
        clock.now = 10 + step
        assert manager.lookup(b"k", req) is api.base.LookupResult.RETRY
        tick(api, manager)
    assert manager.lookup(b"k", req) is api.base.LookupResult.MISS
    # This is an adapter mapping test. The core owns the original absolute
    # deadline and is independently responsible for not restarting it on polls.


def test_actor_preserves_target_observation_clock(api):
    proof = dict(valid_until_monotonic=17.5, observed_at_monotonic=16.5)
    client = N(
        query_ready=lambda *a, **kw: [(N(name="FETCHABLE"), N(name="REMOTE_G2"))],
        query_ready_observations=lambda *a: [proof],
    )
    response = actor_response(
        dict(id=1, kind="query", request_id="r", keys=[b"k"]), client
    )
    assert response["observations"] == [proof]


def test_core_validity_bounds_manager_cache_without_restarting_clock(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    clock = Clock()
    peer = FakeClient(clock=clock)
    manager = DirectGuardManager(config, client=peer, clock=clock)
    req = api.base.ReqContext("r")
    manager.lookup(b"k", req)
    tick(api, manager)
    query = next(r for r in peer.sent if r.kind == "query")
    peer.answer(query, statuses=[["FETCHABLE", "REMOTE_G2"]])
    peer.responses[-1].observations[0]["valid_until_monotonic"] = 10.2
    tick(api, manager)
    assert manager.lookup(b"k", req) is api.base.LookupResult.HIT
    clock.now = 10.201
    assert manager.lookup(b"k", req) is api.base.LookupResult.RETRY


@pytest.mark.parametrize("proof_case", ["missing", "expired", "old_observation"])
def test_remote_positive_requires_current_native_observation(api, config, proof_case):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    clock = Clock()
    peer = FakeClient(clock=clock)
    manager = DirectGuardManager(config, client=peer, clock=clock)
    req = api.base.ReqContext("r")
    manager.lookup(b"k", req)
    tick(api, manager)
    peer.answer(
        next(r for r in peer.sent if r.kind == "query"),
        statuses=[["FETCHABLE", "REMOTE_G2"]],
    )
    if proof_case == "missing":
        peer.responses[-1].observations = [None]
    elif proof_case == "expired":
        peer.responses[-1].observations[0]["valid_until_monotonic"] = 10
    else:
        peer.responses[-1].observations[0]["observed_at_monotonic"] = 4
        peer.responses[-1].observations[0]["valid_until_monotonic"] = 20
    tick(api, manager)
    assert manager.lookup(b"k", req) is api.base.LookupResult.RETRY


def test_native_five_second_expiry_is_not_extended_on_cache_hits(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    clock = Clock()
    peer = FakeClient(clock=clock)
    manager = DirectGuardManager(config, client=peer, clock=clock)
    req = api.base.ReqContext("r")
    manager.lookup(b"k", req)
    tick(api, manager)
    peer.answer(
        next(r for r in peer.sent if r.kind == "query"),
        statuses=[["FETCHABLE", "REMOTE_G2"]],
    )
    tick(api, manager)
    for now in (11.0, 12.0, 14.99):
        clock.now = now
        assert manager.lookup(b"k", req) is api.base.LookupResult.HIT
    clock.now = 15.0
    assert manager.lookup(b"k", req) is api.base.LookupResult.RETRY


@pytest.mark.parametrize(
    "changed_field,new_value",
    [
        ("source_incarnation", "new-source"),
        ("source_progress_epoch", 1),
        ("request_generation", "new-hint"),
    ],
)
def test_new_native_identity_invalidates_remote_cache_and_old_inflight(
    api, config, changed_field, new_value
):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    clock = Clock()
    peer = FakeClient(clock=clock)
    manager = DirectGuardManager(config, client=peer, clock=clock)
    req = api.base.ReqContext("r")
    for k in (b"cached", b"local"):
        manager.lookup(k, req)
    tick(api, manager)
    q = next(r for r in peer.sent if r.kind == "query")
    peer.answer(q, statuses=[["FETCHABLE", "REMOTE_G2"], ["HIT", "LOCAL_G2"]])
    tick(api, manager)
    manager.lookup(b"late", req)
    tick(api, manager)
    late = [r for r in peer.sent if r.kind == "query"][-1]
    manager.lookup(b"new", req)
    tick(api, manager)
    fresh = [r for r in peer.sent if r.kind == "query"][-1]
    peer.answer(late, statuses=[["FETCHABLE", "REMOTE_G2"]])
    old_response = peer.responses.pop()
    clock.now += 0.1
    peer.answer(fresh, statuses=[["FETCHABLE", "REMOTE_G2"]])
    peer.responses[-1].observations[0][changed_field] = new_value
    tick(api, manager)
    assert manager.lookup(b"new", req) is api.base.LookupResult.HIT
    assert manager.lookup(b"cached", req) is api.base.LookupResult.RETRY
    assert manager.lookup(b"local", req) is api.base.LookupResult.HIT
    peer.responses.append(old_response)
    tick(api, manager)
    assert manager.lookup(b"late", req) is api.base.LookupResult.RETRY
    assert (
        "new" in str(manager._requests["r"].remote_identity)
        or changed_field == "source_progress_epoch"
    )


def test_hma_load_retains_hint_after_expiry_and_quarantine(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    clock = Clock()
    cfg, req, groups = geometry(api, config)
    positions(req, groups)
    peer = FakeClient(clock=clock)
    manager = DirectGuardManager(cfg, client=peer, clock=clock)
    for k in (key(64, 0), key(256, 2)):
        manager.lookup(k, req)
    tick(api, manager)
    peer.answer(
        next(r for r in peer.sent if r.kind == "query"),
        statuses=[["FETCHABLE", "REMOTE_G2"]] * 2,
    )
    tick(api, manager)
    assert manager.prepare_load([key(64, 0)], req).hint
    clock.now += 5.001
    assert manager.prepare_load(
        [key(64, 0)], req
    ).hint  # Fresh native GET claim decides availability.
    manager.mark_load_failure(req.req_id)
    assert manager.lookup(key(64, 0), req) is api.base.LookupResult.MISS


def test_real_core_query_ready_actor_wire_and_manager_bridge(api, config, monkeypatch):
    if not (KVCR_SOURCE / "tests/unit/test_query_ready.py").is_file():
        pytest.skip("Set KVCR_SOURCE_DIR to the harryx/guard-direct KVCR checkout")
    # Real opt-in core/progress; byte-address fake NIXL and in-memory control.
    # No CUDA, sockets, native SDK, framework source calls, or package changes.

    monkeypatch.syspath_prepend(str(KVCR_SOURCE / "tests/unit"))
    monkeypatch.syspath_prepend(str(KVCR_SOURCE / "src"))
    from _kvcr_test_utils import _OPEN_KVCRS
    from test_query_ready import exchange, pair

    DirectGuardManager = direct_module("manager").DirectGuardManager

    source, target, left, right, agent, pins, memory = pair()
    try:

        class Peer(FakeClient):
            def flush(self):
                exchange(left, right)
                for request in self.sent:
                    if request.id not in self.pending:
                        continue
                    if request.kind == "query":
                        response = actor_response(
                            dict(
                                id=request.id,
                                kind="query",
                                request_id=request.request_id,
                                keys=request.keys,
                            ),
                            target,
                        )
                        assert "error" not in response
                        self.answer(
                            request,
                            statuses=response["statuses"],
                            observations=response["observations"],
                        )
                    else:
                        self.answer(request)

        peer = Peer()
        manager = DirectGuardManager(config, client=peer)
        req = api.base.ReqContext("req")
        results = None
        until = monotonic() + 2
        while monotonic() < until:
            results = [manager.lookup(k, req) for k in (b"a", b"b")]
            if all(r is not api.base.LookupResult.RETRY for r in results):
                break
            tick(api, manager)
            time.sleep(0.001)
        assert results == [api.base.LookupResult.HIT, api.base.LookupResult.MISS]
        assert (
            manager._requests["req"].remote_identity[2]
            == source._core._remote_fw_dram._dangling_ops.incarnation
        )
        assert source._core._local_dram._export_readers() == 0
        assert not pins.searches
        manager.shutdown()
    finally:
        for client in (target, source):
            client.close()
            if client in _OPEN_KVCRS:
                _OPEN_KVCRS.remove(client)


def test_source_epoch_cannot_roll_backward_even_with_later_observation_clock(
    api, config
):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    clock = Clock()
    peer = FakeClient(clock=clock)
    manager = DirectGuardManager(config, client=peer, clock=clock)
    req = api.base.ReqContext("r")
    for k, epoch in ((b"current", 2), (b"old", 1)):
        manager.lookup(k, req)
        tick(api, manager)
        peer.answer(
            [r for r in peer.sent if r.kind == "query"][-1],
            statuses=[["FETCHABLE", "REMOTE_G2"]],
        )
        peer.responses[-1].observations[0]["source_progress_epoch"] = epoch
        tick(api, manager)
        clock.now += 0.1
    assert manager.lookup(b"current", req) is api.base.LookupResult.HIT
    assert manager.lookup(b"old", req) is api.base.LookupResult.RETRY


@pytest.mark.parametrize("missing", [None, "query_ready", "query_ready_observations"])
def test_rank_proof_requires_both_confirmed_apis_before_reporting_mode(
    api, config, monkeypatch, tmp_path, missing
):
    import json

    DirectIOActor = direct_module("worker").DirectIOActor

    actor = DirectIOActor(
        config,
        N(
            page_bytes=1228800,
            count=9,
            group_ids=(0,),
            region_kwargs=lambda: dict(mem_type="VRAM"),
        ),
    )
    actor.transport_proof = {}
    actor.endpoint = "ipc:///explicit-fixture"
    actor.visibility = N(proof=lambda: dict(visibility="explicit-cpu-fake"))
    actor.client = N(
        _core=N(
            _remote_fw_dram=N(
                _options=N(
                    guard_only_sources=True,
                    guard_only_source_proof=True,
                    opportunistic_query=False,
                )
            ),
            _local_dram=N(_pools={0: ("fixture", 124 * 1024**3)}),
        ),
        query_ready=lambda *a, **kw: [],
        query_ready_observations=lambda *a: [],
    )
    if missing:
        setattr(actor.client, missing, None)
    monkeypatch.setenv("RUN_DIR", str(tmp_path))
    path = tmp_path / "option-a-rank-0.json"
    if missing:
        with pytest.raises(RuntimeError, match="confirmed"):
            actor._proof()
        assert not path.exists()
    else:
        actor._proof()
        proof = json.loads(path.read_text())
        assert proof["remote_query_mode"] == "source_confirmed_ready"
        assert "observation_count" not in proof


def test_new_source_identity_invalidates_cached_misses_too(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    clock = Clock()
    peer = FakeClient(clock=clock)
    manager = DirectGuardManager(config, client=peer, clock=clock)
    req = api.base.ReqContext("r")
    for k in (b"remote", b"missing"):
        manager.lookup(k, req)
    tick(api, manager)
    peer.answer(
        next(r for r in peer.sent if r.kind == "query"),
        statuses=[["FETCHABLE", "REMOTE_G2"], ["MISS", None]],
    )
    tick(api, manager)
    assert manager.lookup(b"missing", req) is api.base.LookupResult.MISS
    clock.now += 0.1
    manager.lookup(b"new", req)
    tick(api, manager)
    peer.answer(
        [r for r in peer.sent if r.kind == "query"][-1],
        statuses=[["FETCHABLE", "REMOTE_G2"]],
    )
    peer.responses[-1].observations[0]["source_incarnation"] = "new-source"
    tick(api, manager)
    assert manager.lookup(b"missing", req) is api.base.LookupResult.RETRY


@pytest.mark.parametrize("parents", [2, 480])
def test_periodic_sparse_mask_hma_retreat_and_actual_load_selection_all_ready(
    api, config, parents
):
    from .test_protocol import ROOT

    DirectGuardManager = direct_module("manager").DirectGuardManager

    sizes, windows = (64, 64, 256, 8, 4), (2, 2, None, 16, 2)
    last = parents * 256
    cfg, req, groups = geometry(
        api, config, parents=tuple(range(256, last + 1, 256)), total=last + 4
    )
    positions(req, groups)
    ready: set[bytes] = set()
    for end in range(256, last + 1, 256):
        for g, ends in {
            0: (end - 64, end),
            1: (end - 64, end, end + 64),
            2: (end,),
            3: range(end - 120, end + 1, 8),
            4: (end - 4, end),
        }.items():
            ready.update(key(e, g) for e in ends)
    peer = ConfirmedPeer(ready)
    manager = DirectGuardManager(cfg, client=peer)
    manager.on_new_request(req)
    hma = pinned_hma(api)()
    hma.manager = manager
    hma._lookup_groups = (2, 0, 1, 3, 4)
    hma._sliding_window_groups = (0, 1, 3, 4)
    hma._mamba_align_size = None
    hma._chunks_being_loaded = set()
    hma._events_tracker = N(record_lookup=lambda *args: None)
    configs = [
        N(
            tokens_per_chunk=size,
            tokens_per_block=size,
            group_idx=g,
            sliding_window_size_in_chunks=w,
            is_eagle_group=g == 1,
            kv_cache_spec=N(),
            load_window_size_in_chunks=lambda end, w=w: w,
        )
        for g, (size, w) in enumerate(zip(sizes, windows))
    ]
    hma.config = N(kv_group_configs=configs, num_workers=1, blocks_per_chunk=1)
    state = N(
        num_locally_computed_tokens=0,
        max_load_tokens=None,
        req=N(num_tokens=last + 4, request_id=req.req_id),
        req_context=req,
        group_states=[N(offload_keys=ks) for ks in groups],
        partial_tail_boundary=None,
        load_key_ranges=[],
        transfer_jobs=set(),
        offloading_context=api.base.RequestOffloadingContext(),
    )
    # Reproduce unsafe r5 boundary with eligibility-only metadata first.
    real_manager = hma.manager
    hma.manager = N(lookup=lambda *args: api.base.LookupResult.HIT)
    assert hma._lookup_complete_chunks(state) == last - 64
    hma.manager = real_manager
    for _ in range(12):
        hit = hma._lookup_complete_chunks(state)
        if hit is not None:
            break
        tick(api, manager)
    assert hit == last - 256 and hit > 0
    # Execute the exact installed adapter scheduler's load-key selection body.
    path = ROOT / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "OffloadingConnectorScheduler"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "update_state_after_alloc"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    ns = dict(
        cdiv=lambda a, b: -(-a // b),
        MambaSpec=type("ExplicitFakeMamba", (), {}),
        OffloadPolicy=api.base.OffloadPolicy,
        GPULoadStoreSpec=api.base.GPULoadStoreSpec,
        TransferJob=lambda **kwargs: N(**kwargs),
        TransferJobStatus=lambda **kwargs: N(**kwargs),
    )
    exec(compile(module, str(path), "exec"), ns)
    allocated = []
    for g, (size, window) in enumerate(zip(sizes, windows)):
        count = -(-hit // size)
        live = count if window is None else window
        allocated.append(
            [N(block_id=0, is_null=True, block_hash=None) for _ in range(count - live)]
            + [
                N(block_id=g * 100000 + i + 1, is_null=False, block_hash=None)
                for i in range(count - live, count)
            ]
        )
    hma._req_status = {req.req_id: state}
    hma._current_batch_allocated_block_ids = set()
    hma._current_batch_load_jobs = {}
    hma._jobs = {}
    hma._generate_job_id = lambda: 99
    ns["update_state_after_alloc"](hma, state.req, N(blocks=allocated), hit)
    selected = hma._current_batch_load_jobs[99].src_spec.keys
    assert len(selected) == (parents - 1) + 22
    assert set(selected).issubset(ready)
    assert all(manager.lookup(k, req) is api.base.LookupResult.HIT for k in selected)
    assert hma._current_batch_load_jobs[99].dst_spec.group_sizes == [
        2,
        2,
        parents - 1,
        16,
        2,
    ]
