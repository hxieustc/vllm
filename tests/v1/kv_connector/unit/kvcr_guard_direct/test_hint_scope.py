# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real pinned context and HMA methods, explicit CPU-only worker peer."""

import ast
import copy
import dataclasses
import hashlib
import logging
from itertools import islice
from types import SimpleNamespace

import pytest

from ._paths import ROOT, direct_module
from .test_manager import FakeClient, tick

SIZES = (64, 64, 256, 8, 4)


def key(end, group):
    return hashlib.sha256(str(end).encode()).digest() + group.to_bytes(4, "big")


def external(k):
    return int.from_bytes(k[:-4], "big") & ((1 << 64) - 1)


def envelope(parents=(256,)):
    return dict(
        protocol_version="0.1",
        message_id="original",
        actions=[
            dict(
                action_id="fetch-original",
                action_type="kv.fetch",
                action_version="1.0",
                payload=dict(
                    source_control_endpoint="tcp://source:17771",
                    mode="copy",
                    block_hashes=[external(key(end, 2)) for end in parents],
                    other="preserved",
                ),
            )
        ],
    )


def geometry(api, config, parents=(256,), total=256):
    cfg = dataclasses.replace(
        config,
        groups=tuple(
            api.config.OffloadingGroupConfig(size, (f"layer{g}",), g)
            for g, size in enumerate(SIZES)
        ),
        cache=api.config.OffloadingCacheConfig(4, 1),
        extra_config={**config.extra_config, "router_hint_block_size": 256},
    )
    req = api.base.ReqContext("hinted", kv_hints=envelope(parents))
    groups = [
        [key(end, g) for end in range(size, total + 1, size)]
        for g, size in enumerate(SIZES)
    ]
    return cfg, req, groups


class StrictPeer(FakeClient):
    def __init__(self):
        super().__init__()
        self.hints = {}

    def flush(self):
        for request in self.sent:
            if request.id not in self.pending:
                continue
            if request.kind == "hint":
                self.hints[request.request_id] = request.hint
            hashes = set(
                self.hints.get(request.request_id, {})
                .get("actions", [{}])[0]
                .get("payload", {})
                .get("block_hashes", [])
            )
            self.answer(
                request,
                statuses=[
                    ["FETCHABLE", "REMOTE_G2"]
                    if external(k) in hashes
                    else ["MISS", None]
                    for k in request.keys
                ]
                if request.kind == "query"
                else [],
            )


def positions(req, groups):
    for g, keys in enumerate(groups):
        for i, k in enumerate(keys, 1):
            req.set_offload_key_position(k, i * SIZES[g])


def test_105_fine_keys_normalized_at_flush_and_reused_for_load(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    cfg, req, groups = geometry(api, config)
    raw = copy.deepcopy(req.kv_hints)
    peer = StrictPeer()
    m = DirectGuardManager(cfg, client=peer)
    m.on_new_request(req)  # Positions arrive later, as in the pinned scheduler.
    positions(req, groups)
    keys: list[bytes] = sum(groups, [])
    assert len(keys) == 105
    assert (
        sum(external(k) in raw["actions"][0]["payload"]["block_hashes"] for k in keys)
        == 5
    )
    for k in keys:
        assert m.lookup(k, req) is api.base.LookupResult.RETRY
    tick(api, m)
    assert all(m.lookup(k, req) is api.base.LookupResult.HIT for k in keys)
    assert req.kv_hints == raw
    normalized = next(r.hint for r in peer.sent if r.kind == "hint")
    assert normalized != raw and m.prepare_load(keys, req).hint == normalized
    assert (
        normalized["actions"][0]["payload"]["source_control_endpoint"]
        == "tcp://source:17771"
    )
    assert normalized["actions"][0]["payload"]["other"] == "preserved"


@pytest.mark.parametrize("bad", [None, True, 0, 64, 128, "256"])
def test_real_fetch_hint_requires_explicit_certified_block_size(api, config, bad):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    cfg, req, _ = geometry(api, config)
    extra = dict(cfg.extra_config)
    if bad is None:
        extra.pop("router_hint_block_size")
    else:
        extra["router_hint_block_size"] = bad
    with pytest.raises(ValueError, match="router_hint_block_size"):
        DirectGuardManager(
            dataclasses.replace(cfg, extra_config=extra), client=StrictPeer()
        ).on_new_request(req)


@pytest.mark.parametrize(
    "case", ["missing_parent", "conflicting_parent", "missing_position", "tail", "hole"]
)
def test_unverified_scope_stays_miss(api, config, case):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    cfg, req, groups = geometry(api, config, total=512)
    m = DirectGuardManager(cfg, client=StrictPeer())
    m.on_new_request(req)
    positions(req, groups)
    target = key(64, 0)
    candidates = [target, key(256, 2)]
    if case == "missing_parent":
        candidates = [target]
    if case == "conflicting_parent":
        conflict = b"different" + (2).to_bytes(4, "big")
        req.set_offload_key_position(conflict, 256)
        candidates.append(conflict)
    if case == "missing_position":
        target = key(64, 0) + b"xxxx"
        candidates = [target, key(256, 2)]
    if case in ("tail", "hole"):
        target = key(320, 0)
        candidates = [target, key(256, 2), key(512, 2)]
    for k in candidates:
        m.lookup(k, req)
    tick(api, m)
    assert m.lookup(target, req) is api.base.LookupResult.MISS


def test_geometry_queries_reproduce_2165_of_45466_before_normalization(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    # 433 complete canonical parents and an unhinted partial final parent.
    cfg, req, groups = geometry(
        api, config, tuple(range(256, 110849, 256)), total=110852
    )
    keys: list[bytes] = sum(groups, [])
    assert len(keys) == 45466
    coarse = set(req.kv_hints["actions"][0]["payload"]["block_hashes"])
    assert sum(external(k) in coarse for k in keys) == 2165
    peer = StrictPeer()
    m = DirectGuardManager(cfg, client=peer)
    m.on_new_request(req)
    positions(req, groups)
    for k in keys:
        m.lookup(k, req)
    while m._requests[req.req_id].queued:
        tick(api, m)
    for g, ks in enumerate(groups):
        for i, k in enumerate(ks, 1):
            expected = (
                api.base.LookupResult.HIT
                if i * SIZES[g] <= 110848
                else api.base.LookupResult.MISS
            )
            assert m.lookup(k, req) is expected


def test_attestation_is_bounded_counts_and_digests(api, config, caplog):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    cfg, req, groups = geometry(api, config)
    m = DirectGuardManager(cfg, client=StrictPeer())
    m.on_new_request(req)
    positions(req, groups)
    with caplog.at_level(logging.INFO):
        for k in sum(groups, []):
            m.lookup(k, req)
        tick(api, m)
    proof = m.get_hint_scope_attestation(req.req_id)
    assert proof["candidate_keys"] == 105 and proof["authorized_keys"] == 105
    assert proof["original_parent_hashes"] == 1
    assert len(proof["normalized_hint_sha256"]) == 64
    assert "KVCR_HINT_SCOPE" in caplog.text
    assert (
        "source_control_endpoint" not in caplog.text
        and key(64, 0).hex() not in caplog.text
    )
    m.mark_load_failure(req.req_id)
    assert m.lookup(key(64, 0), req) is api.base.LookupResult.MISS
    m.on_request_finished(req)
    assert m.get_hint_scope_attestation(req.req_id) is None


def pinned_hma(api):
    """Execute the exact pinned HMA methods; fake only unrelated imports/types."""
    path = ROOT / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "OffloadingConnectorScheduler"
    )
    names = {
        "_maximal_prefix_lookup",
        "_sliding_window_lookup",
        "_lookup_complete_chunks",
    }
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    ns = dict(
        LookupResult=api.base.LookupResult,
        cdiv=lambda a, b: -(-a // b),
        round_down=lambda a, b: a // b * b,
        logger=logging.getLogger("pinned-hma"),
    )
    exec(compile(module, str(path), "exec"), ns)
    return ns["OffloadingConnectorScheduler"]


def test_actual_pinned_hma_converges_across_five_groups(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    cfg, req, groups = geometry(api, config, parents=(256, 512), total=512)
    peer = StrictPeer()
    m = DirectGuardManager(cfg, client=peer)
    m.on_new_request(req)
    positions(req, groups)
    scheduler = pinned_hma(api)()
    scheduler.manager = m
    scheduler._lookup_groups = (2, 0, 1, 3, 4)
    scheduler._sliding_window_groups = (0, 1, 3, 4)
    scheduler._mamba_align_size = None
    scheduler._chunks_being_loaded = set()
    scheduler._events_tracker = SimpleNamespace(record_lookup=lambda *args: None)
    configs = [
        SimpleNamespace(
            tokens_per_chunk=size,
            sliding_window_size_in_chunks=None if g == 2 else 2,
            is_eagle_group=g == 1,
            load_window_size_in_chunks=lambda end: 2,
        )
        for g, size in enumerate(SIZES)
    ]
    scheduler.config = SimpleNamespace(kv_group_configs=configs)
    state = SimpleNamespace(
        num_locally_computed_tokens=0,
        max_load_tokens=None,
        req=SimpleNamespace(num_tokens=513, request_id=req.req_id),
        req_context=req,
        group_states=[SimpleNamespace(offload_keys=ks) for ks in groups],
    )
    coarse = set(req.kv_hints["actions"][0]["payload"]["block_hashes"])
    scheduler.manager = SimpleNamespace(
        lookup=lambda k, ctx: api.base.LookupResult.HIT
        if external(k) in coarse
        else api.base.LookupResult.MISS
    )
    assert scheduler._lookup_complete_chunks(state) == 0
    scheduler.manager = m
    # The current strict peer with coarse hints cannot supply consecutive windows.
    assert scheduler._lookup_complete_chunks(state) is None
    tick(api, m)
    assert m.lookup(key(64, 0), req) is api.base.LookupResult.HIT
    assert (
        scheduler._lookup_complete_chunks(state) == 448
    )  # Pinned EAGLE drops its trailing 64-token chunk.


def test_fine_boundary_alone_is_not_a_canonical_anchor(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    cfg, req, groups = geometry(api, config)
    positions(req, groups)
    m = DirectGuardManager(cfg, client=StrictPeer())
    m.on_new_request(req)
    for k in (key(64, 0), key(256, 0)):
        m.lookup(k, req)
    tick(api, m)
    assert m.lookup(key(64, 0), req) is api.base.LookupResult.MISS


def test_sparse_parents_do_not_fill_holes_or_authorize_trailing_partial_parent(
    api, config
):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    cfg, req, groups = geometry(api, config, parents=(256, 768), total=832)
    positions(req, groups)
    m = DirectGuardManager(cfg, client=StrictPeer())
    m.on_new_request(req)
    for k in sum(groups, []):
        m.lookup(k, req)
    tick(api, m)
    for end, expected in (
        (64, True),
        (256, True),
        (320, False),
        (512, False),
        (576, True),
        (768, True),
        (832, False),
    ):
        assert (m.lookup(key(end, 0), req) is api.base.LookupResult.HIT) == expected


def test_failed_hint_initialization_cannot_leave_unvalidated_raw_hint_active(
    api, config
):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    cfg, req, _ = geometry(api, config)
    extra = dict(cfg.extra_config)
    extra.pop("router_hint_block_size")
    peer = StrictPeer()
    m = DirectGuardManager(dataclasses.replace(cfg, extra_config=extra), client=peer)
    with pytest.raises(ValueError):
        m.on_new_request(req)
    with pytest.raises(ValueError):
        m.lookup(key(64, 0), req)
    assert not peer.sent


def test_late_parent_conflict_revokes_cached_remote_hit_before_job_preparation(
    api, config
):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    cfg, req, groups = geometry(api, config)
    positions(req, groups)
    m = DirectGuardManager(cfg, client=StrictPeer())
    m.on_new_request(req)
    for k in sum(groups, []):
        m.lookup(k, req)
    tick(api, m)
    assert m.lookup(key(64, 0), req) is api.base.LookupResult.HIT
    conflict = b"conflict" + (2).to_bytes(4, "big")
    req.set_offload_key_position(conflict, 256)
    m.lookup(conflict, req)
    assert m.lookup(key(64, 0), req) is api.base.LookupResult.MISS
    with pytest.raises(ValueError):
        m.prepare_load([key(64, 0)], req)


def test_hinted_request_bounds_and_generation_identity(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    cfg, req, groups = geometry(api, config)
    positions(req, groups)
    cfg = dataclasses.replace(
        cfg, extra_config={**cfg.extra_config, "metadata_max_keys_per_request": 2}
    )
    c = FakeClient()
    m = DirectGuardManager(cfg, client=c)
    m.on_new_request(req)
    for k in (key(64, 0), key(256, 2)):
        m.lookup(k, req)
    with pytest.raises(BufferError):
        m.lookup(key(128, 0), req)
    assert len(m._requests[req.req_id].hint_scope.candidates) == 2
    tick(api, m)
    old = next(r for r in c.sent if r.kind == "query")
    m.on_request_finished(req)
    new = api.base.ReqContext(req.req_id, kv_hints=envelope((512,)))
    for end, group in ((320, 0), (512, 2)):
        new.set_offload_key_position(key(end, group), end)
    m.on_new_request(new)
    for k in (key(320, 0), key(512, 2)):
        m.lookup(k, new)
    c.answer(old, statuses=[["FETCHABLE", "REMOTE_G2"]] * 2)
    tick(api, m)
    assert key(64, 0) not in m._requests[req.req_id].statuses
    hashes = m.prepare_load([key(320, 0)], new).hint["actions"][0]["payload"][
        "block_hashes"
    ]
    assert external(key(512, 2)) in hashes and external(key(256, 2)) not in hashes
    m.mark_load_failure(new.req_id)
    query = [r for r in c.sent if r.kind == "query"][-1]
    c.answer(query, statuses=[["FETCHABLE", "REMOTE_G2"]] * 2)
    tick(api, m)
    assert m.lookup(key(320, 0), new) is api.base.LookupResult.MISS


@pytest.mark.parametrize("value", [True, 64, 128, "256"])
def test_spec_rejects_invalid_explicit_hint_contract(api, config, value):
    KVCRDirectGuardSpec = direct_module("spec").KVCRDirectGuardSpec

    with pytest.raises(ValueError, match="router_hint_block_size"):
        KVCRDirectGuardSpec(
            dataclasses.replace(
                config,
                extra_config={**config.extra_config, "router_hint_block_size": value},
            )
        )


def test_hint_contract_requires_canonical_group_not_just_fine_aligned_keys(api, config):
    KVCRDirectGuardSpec = direct_module("spec").KVCRDirectGuardSpec

    cfg, _, _ = geometry(api, config)
    with pytest.raises(ValueError, match="LCM"):
        KVCRDirectGuardSpec(dataclasses.replace(cfg, groups=cfg.groups[:2]))


def test_actual_pinned_key_producer_populates_context_before_normalization(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    cfg, req, expected = geometry(api, config)
    m = DirectGuardManager(cfg, client=StrictPeer())
    m.on_new_request(req)
    path = ROOT / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "RequestOffloadState"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "update_offload_keys"
    )
    ns = dict(islice=islice, make_offload_key=api.base.make_offload_key)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
    groups = [SimpleNamespace(offload_keys=[]) for _ in SIZES]
    state = SimpleNamespace(
        config=SimpleNamespace(
            kv_group_configs=[
                SimpleNamespace(
                    hashes_per_chunk=size // 4, group_idx=g, tokens_per_chunk=size
                )
                for g, size in enumerate(SIZES)
            ]
        ),
        req=SimpleNamespace(
            block_hashes=[key(end, 0)[:-4] for end in range(4, 257, 4)]
        ),
        req_context=req,
        group_states=groups,
    )
    ns["update_offload_keys"](state)
    assert [g.offload_keys for g in groups] == expected
    for g in groups:
        for k in g.offload_keys:
            m.lookup(k, req)
    tick(api, m)
    assert all(
        m.lookup(k, req) is api.base.LookupResult.HIT
        for g in groups
        for k in g.offload_keys
    )


@pytest.mark.parametrize("old_status", [["MISS", None], ["FETCHABLE", "REMOTE_G2"]])
def test_late_scope_expansion_discards_inflight_snapshot_and_requeries(
    api, config, old_status
):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    cfg, req, groups = geometry(api, config)
    positions(req, groups)
    peer = FakeClient()
    m = DirectGuardManager(cfg, client=peer)
    m.on_new_request(req)
    fine, parent = key(64, 0), key(256, 2)
    assert m.lookup(fine, req) is api.base.LookupResult.RETRY
    tick(api, m)
    old = next(r for r in peer.sent if r.kind == "query")
    assert m.lookup(parent, req) is api.base.LookupResult.RETRY
    tick(api, m)
    canonical = [r for r in peer.sent if r.kind == "query"][-1]
    peer.answer(old, statuses=[old_status])
    peer.answer(canonical, statuses=[["FETCHABLE", "REMOTE_G2"]])
    tick(api, m)
    assert m.lookup(parent, req) is api.base.LookupResult.HIT
    assert m.lookup(fine, req) is api.base.LookupResult.RETRY
    queries = [r for r in peer.sent if r.kind == "query" and fine in r.keys]
    assert len(queries) == 2
    normalized = [r for r in peer.sent if r.kind == "hint"][-1]
    assert normalized.id < queries[-1].id
    assert external(fine) in normalized.hint["actions"][0]["payload"]["block_hashes"]
    peer.answer(queries[-1], statuses=[["FETCHABLE", "REMOTE_G2"]])
    tick(api, m)
    assert m.lookup(fine, req) is api.base.LookupResult.HIT
    assert m.prepare_load([fine], req).hint == normalized.hint


@pytest.mark.parametrize("barrier", ["quarantine", "finish", "inventory", "conflict"])
def test_deferred_scope_requery_preserves_failure_generation_and_epoch(
    api, config, barrier
):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    cfg, req, groups = geometry(api, config)
    positions(req, groups)
    peer = FakeClient()
    m = DirectGuardManager(cfg, client=peer)
    m.on_new_request(req)
    fine = key(64, 0)
    m.lookup(fine, req)
    tick(api, m)
    old = next(r for r in peer.sent if r.kind == "query")
    m.lookup(key(256, 2), req)
    tick(api, m)
    if barrier == "quarantine":
        m.mark_load_failure(req.req_id)
    elif barrier == "finish":
        m.on_request_finished(req)
        req = api.base.ReqContext(req.req_id, kv_hints=envelope())
        m.on_new_request(req)
    elif barrier == "inventory":
        poll = next(r for r in peer.sent if r.kind == "poll")
        peer.answer(poll, events=[dict(keys=[fine], removed=True, tier="LOCAL_G2")])
        tick(api, m)
    else:
        conflict = b"conflict" + (2).to_bytes(4, "big")
        req.set_offload_key_position(conflict, 256)
        m.lookup(conflict, req)
        tick(api, m)
    peer.answer(old, statuses=[["FETCHABLE", "REMOTE_G2"]])
    tick(api, m)
    assert fine not in m._requests[req.req_id].statuses
    queries = [r for r in peer.sent if r.kind == "query" and fine in r.keys]
    if barrier in ("quarantine", "finish"):
        assert len(queries) == 1
    else:
        assert len(queries) == 2
        peer.answer(queries[-1], statuses=[["FETCHABLE", "REMOTE_G2"]])
        tick(api, m)
        expected = (
            api.base.LookupResult.MISS
            if barrier == "conflict"
            else api.base.LookupResult.HIT
        )
        assert m.lookup(fine, req) is expected


def test_deferred_requery_at_final_drain_is_bounded_and_survives_backpressure(
    api, config, monkeypatch
):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    cfg, req, groups = geometry(api, config)
    positions(req, groups)
    cfg = dataclasses.replace(
        cfg, extra_config={**cfg.extra_config, "metadata_max_keys_per_request": 2}
    )
    peer = FakeClient()
    m = DirectGuardManager(cfg, client=peer)
    m.on_new_request(req)
    fine, parent = key(64, 0), key(256, 2)
    m.lookup(fine, req)
    tick(api, m)
    old = next(r for r in peer.sent if r.kind == "query")
    m.lookup(parent, req)

    def deliver_during_flush():
        monkeypatch.setattr(peer, "flush", lambda: None)
        peer.answer(old, statuses=[["MISS", None]])

    monkeypatch.setattr(peer, "flush", deliver_during_flush)
    tick(api, m)  # Reply arrives at the final drain, after authorization flush.
    state = m._requests[req.req_id]
    assert fine in state.queued and not state.deferred_requery
    assert len(state.known) == 2 and m.has_pending_work()
    assert m.lookup(fine, req) is api.base.LookupResult.RETRY
    with pytest.raises(BufferError):
        m.lookup(key(128, 0), req)
    request = peer.request

    def backpressure(kind, *args, **kwargs):
        if kind == "query":
            raise BufferError("peer full")
        return request(kind, *args, **kwargs)

    monkeypatch.setattr(peer, "request", backpressure)
    tick(api, m)
    assert fine in state.queued
    monkeypatch.setattr(peer, "request", request)
    tick(api, m)
    fresh = [r for r in peer.sent if r.kind == "query" and fine in r.keys][-1]
    assert fresh.id != old.id
    peer.answer(fresh, statuses=[["FETCHABLE", "REMOTE_G2"]])
    tick(api, m)
    assert m.lookup(fine, req) is api.base.LookupResult.HIT
    assert len(state.known) == 2 and not state.deferred_requery
    assert sum(fine in r.keys for r in peer.sent if r.kind == "query") == 2
