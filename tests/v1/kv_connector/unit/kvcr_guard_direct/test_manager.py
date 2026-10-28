# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
from time import monotonic

import msgspec
import pytest

from ._paths import direct_module


class FakeClient:
    """Asynchronous metadata peer; no native KVCR, torch, CUDA, or data tier."""

    def __init__(self, clock=None):
        self.sent = []
        self.responses = []
        self.pending = set()
        self.closed = False
        self.clock = clock if clock is not None else monotonic

    def request(self, kind, request_id="", keys=(), hint=None):
        Request = direct_module("protocol").Request

        rid = len(self.sent)
        self.sent.append(Request(rid, kind, request_id, list(keys), hint))
        self.pending.add(rid)
        return rid

    def flush(self):
        pass

    def drain(self):
        rows, self.responses = self.responses, []
        for r in rows:
            self.pending.remove(r.id)
        return rows

    def has_pending_work(self):
        return bool(self.pending)

    def close(self):
        self.closed = True

    def answer(self, request, **kwargs):
        Response = direct_module("protocol").Response

        if "observations" not in kwargs and kwargs.get("statuses"):
            now = self.clock()
            kwargs["observations"] = [
                dict(
                    status=pair[0],
                    observed_at_monotonic=now,
                    valid_until_monotonic=now + 5,
                    request_generation="fake-hint-generation",
                    source_endpoint="tcp://source:17771",
                    source_incarnation="fake-source",
                    source_progress_epoch=0,
                    residency_generation=1,
                )
                if pair == ["FETCHABLE", "REMOTE_G2"] or pair == ["HIT", "REMOTE_G2"]
                else None
                for pair in kwargs["statuses"]
            ]
        self.responses.append(Response(request.id, **kwargs))


@pytest.fixture
def manager(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    client = FakeClient()
    m = DirectGuardManager(config, client=client)
    return m, client


def tick(api, m):
    m.on_schedule_end(api.base.ScheduleEndContext((), ()))


def test_inventory_gate_skips_unhinted_absent_keys_and_queries_ready_keys(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    cfg = dataclasses.replace(
        config, extra_config={**config.extra_config, "metadata_inventory_gate": True}
    )
    client = FakeClient()
    manager = DirectGuardManager(cfg, client=client)
    req = api.base.ReqContext("unhinted")
    assert manager.lookup(b"present-later", req) is api.base.LookupResult.MISS
    tick(api, manager)
    assert not any(row.kind == "query" for row in client.sent)
    poll = next(row for row in client.sent if row.kind == "poll")
    client.answer(
        poll, events=[dict(tier="LOCAL_G2", keys=[b"present-later"], removed=False)]
    )
    tick(api, manager)
    assert manager.lookup(b"present-later", req) is api.base.LookupResult.RETRY
    tick(api, manager)
    assert [row.keys for row in client.sent if row.kind == "query"] == [
        [b"present-later"]
    ]
    query = next(row for row in client.sent if row.kind == "query")
    client.answer(query, statuses=[["HIT", "LOCAL_G2"]])
    tick(api, manager)
    assert manager.lookup(b"present-later", req) is api.base.LookupResult.HIT
    poll = [
        row for row in client.sent if row.kind == "poll" and row.id in client.pending
    ][-1]
    client.answer(
        poll, events=[dict(tier="LOCAL_G2", keys=[b"present-later"], removed=True)]
    )
    tick(api, manager)
    assert manager.lookup(b"present-later", req) is api.base.LookupResult.MISS


def test_queries_batch_per_request_and_pending_fetch_is_queried_again(api, manager):
    m, c = manager
    a = api.base.ReqContext("a")
    b = api.base.ReqContext("b")
    for key in (b"1", b"2", b"1"):
        assert m.lookup(key, a) is api.base.LookupResult.RETRY
    assert m.lookup(b"1", b) is api.base.LookupResult.RETRY
    assert not c.sent
    tick(api, m)
    qs = [r for r in c.sent if r.kind == "query"]
    assert [(r.request_id, r.keys) for r in qs] == [("a", [b"1", b"2"]), ("b", [b"1"])]
    c.answer(qs[0], statuses=[["FETCHING", "REMOTE_G2"], ["FETCHABLE", "REMOTE_G2"]])
    c.answer(qs[1], statuses=[["MISS", None]])
    tick(api, m)
    assert m.lookup(b"1", a) is api.base.LookupResult.RETRY
    assert m.lookup(b"2", a) is api.base.LookupResult.HIT
    assert m.lookup(b"1", b) is api.base.LookupResult.MISS
    tick(api, m)
    q = [r for r in c.sent if r.kind == "query"][-1]
    assert q.keys == [b"1"] and q.request_id == "a"
    c.answer(q, statuses=[["HIT", "LOCAL_G2"]])
    tick(api, m)
    assert m.lookup(b"1", a) is api.base.LookupResult.HIT


def test_long_request_queries_are_chunked_without_losing_queued_keys(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    cfg = dataclasses.replace(
        config,
        extra_config={
            **config.extra_config,
            "metadata_max_keys_per_request": 8,
            "metadata_query_batch_keys": 2,
        },
    )
    client = FakeClient()
    manager = DirectGuardManager(cfg, client=client)
    req = api.base.ReqContext("long")
    keys = [str(i).encode() for i in range(5)]
    for key in keys:
        assert manager.lookup(key, req) is api.base.LookupResult.RETRY
    for _ in range(3):
        tick(api, manager)
    queries = [row for row in client.sent if row.kind == "query"]
    assert [row.keys for row in queries] == [keys[:2], keys[2:4], keys[4:]]
    assert not manager._requests["long"].queued
    assert manager._requests["long"].inflight == set(keys)
    for row in queries:
        client.answer(row, statuses=[["HIT", "LOCAL_G2"] for _ in row.keys])
    tick(api, manager)
    assert all(manager.lookup(key, req) is api.base.LookupResult.HIT for key in keys)


def test_store_offers_all_keys_without_staging_or_synthetic_inventory(api, manager):
    DirectSpec = direct_module("protocol").DirectSpec

    m, _ = manager
    req = api.base.ReqContext("r")
    keys = [b"a", b"b", b"c"]
    result = m.prepare_store(keys, req)
    assert result.keys_to_store == keys and result.evicted_keys == []
    assert result.store_spec == DirectSpec(tuple(keys), "r")
    assert m.prepare_load(keys, req) == DirectSpec(tuple(keys), "r")
    m.complete_store(keys, req, True)
    m.complete_load(keys, req)
    assert list(m.take_events()) == []
    assert m.lookup(b"a", req) is api.base.LookupResult.RETRY


def test_load_carries_hint_builtins_for_actor_resubmission(api, manager):
    class Hint(msgspec.Struct):
        version: int
        targets: list[str]

    m, _ = manager
    req = api.base.ReqContext("r", kv_hints=Hint(1, ["source"]))
    load = m.prepare_load([b"a"], req)
    assert load.hint == msgspec.to_builtins(req.kv_hints)
    assert m.prepare_store([b"a"], req).store_spec.hint is None


def test_hints_are_exact_builtins_and_finish_cleans_request_but_not_native_jobs(
    api, manager
):
    class Hint(msgspec.Struct):
        version: int
        keys: list[bytes]

    m, c = manager
    req = api.base.ReqContext("r", kv_hints=Hint(1, [b"a"]))
    m.on_new_request(req)
    m.lookup(b"a", req)
    m.touch([b"a", b"b"], req)
    tick(api, m)
    assert [r for r in c.sent if r.kind == "hint"][0].hint == msgspec.to_builtins(
        req.kv_hints
    )
    assert [r for r in c.sent if r.kind == "touch"][0].keys == [b"a", b"b"]
    old = next(r for r in c.sent if r.kind == "query")
    m.on_request_finished(req)
    tick(api, m)
    assert any(r.kind == "finish" and r.request_id == "r" for r in c.sent)
    c.answer(old, statuses=[["HIT", "LOCAL_G2"]])
    tick(api, m)
    assert m.lookup(b"a", api.base.ReqContext("r")) is api.base.LookupResult.RETRY


def test_authoritative_withdrawal_invalidates_cache_and_stale_query(api, manager):
    m, c = manager
    req = api.base.ReqContext("r")
    m.lookup(b"a", req)
    tick(api, m)
    q = next(r for r in c.sent if r.kind == "query")
    poll = next(r for r in c.sent if r.kind == "poll")
    c.answer(poll, events=[{"keys": [b"a"], "removed": True, "tier": "LOCAL_G2"}])
    tick(api, m)
    c.answer(q, statuses=[["HIT", "LOCAL_G2"]])
    tick(api, m)
    assert m.lookup(b"a", req) is api.base.LookupResult.RETRY
    events = list(m.take_events())
    assert len(events) == 1 and events[0].removed and events[0].ownership == "kvcr"
    assert events[0].medium is api.base.Medium.CPU and events[0].removal_expected


def test_inventory_g3_translation_and_disabled_event_capture(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    c = FakeClient()
    m = DirectGuardManager(config, client=c)
    tick(api, m)
    c.answer(c.sent[0], events=[{"keys": [b"k"], "removed": False, "tier": "G3"}])
    tick(api, m)
    assert list(m.take_events())[0].medium is api.base.Medium.STORAGE
    c = FakeClient()
    m = DirectGuardManager(
        dataclasses.replace(config, enable_kv_cache_events=False), client=c
    )
    tick(api, m)
    c.answer(c.sent[0], events=[{"keys": [b"k"], "removed": False, "tier": "LOCAL_G2"}])
    tick(api, m)
    assert list(m.take_events()) == []


def test_typed_stats_use_actual_pinned_stats_with_tuple_labels(api, manager):
    m, c = manager
    tick(api, m)
    c.answer(
        c.sent[0],
        stats=[
            ["kvcr_transfer_bytes", "counter", ["deliver"], 64],
            ["kvcr_duration_seconds", "histogram", ["deliver", "success"], [0.1, 0.2]],
            ["kvcr_state", "gauge", ["pinned_keys"], 3],
        ],
    )
    tick(api, m)
    stats = m.get_stats()
    assert isinstance(stats, api.stats.OffloadingConnectorStats)
    assert stats.data["data"]["vllm:kvcr_transfer_bytes"] == {("deliver",): 64}
    assert stats.data["data"]["vllm:kvcr_duration_seconds"] == {
        ("deliver", "success"): [0.1, 0.2]
    }
    assert stats.data["data"]["vllm:kvcr_state"] == {("pinned_keys",): 3}


def test_malformed_status_count_and_worker_errors_fail_explicitly(api, manager):
    m, c = manager
    m.lookup(b"a", api.base.ReqContext("r"))
    tick(api, m)
    c.answer(next(r for r in c.sent if r.kind == "query"), statuses=[])
    with pytest.raises(ValueError, match="status"):
        tick(api, m)


def test_request_and_key_metadata_have_explicit_bounds(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    cfg = dataclasses.replace(
        config,
        extra_config={
            **config.extra_config,
            "metadata_max_requests": 1,
            "metadata_max_keys_per_request": 1,
        },
    )
    m = DirectGuardManager(cfg, client=FakeClient())
    a = api.base.ReqContext("a")
    m.lookup(b"a", a)
    with pytest.raises(BufferError):
        m.lookup(b"b", a)
    with pytest.raises(BufferError):
        m.lookup(b"a", api.base.ReqContext("b"))
    m.on_request_finished(a)
    assert m.lookup(b"a", api.base.ReqContext("b")) is api.base.LookupResult.RETRY


def test_idle_poll_does_not_keep_engine_alive_but_finish_ack_does(api, manager):
    m, c = manager
    tick(api, m)
    assert not m.has_pending_work()
    req = api.base.ReqContext("r")
    m.on_new_request(req)
    m.on_request_finished(req)
    tick(api, m)
    assert m.has_pending_work()
    finish = next(r for r in c.sent if r.kind == "finish")
    c.answer(finish)
    tick(api, m)
    assert not m.has_pending_work()


def test_worker_error_is_not_a_metadata_miss(api, manager):
    m, c = manager
    tick(api, m)
    c.answer(c.sent[0], error="Guard unavailable")
    with pytest.raises(RuntimeError, match="Guard unavailable"):
        tick(api, m)


def test_worker_error_preserves_authoritative_inventory_and_stats(api, manager):
    m, c = manager
    tick(api, m)
    c.answer(
        c.sent[0],
        error="query failed",
        events=[{"keys": [b"a"], "removed": True, "tier": "LOCAL_G2"}],
        stats=[["kvcr_transfer_bytes", "counter", ["deliver"], 64]],
    )
    with pytest.raises(RuntimeError, match="query failed"):
        tick(api, m)
    assert list(m.take_events())[0].removed
    assert m.get_stats().data["data"]["vllm:kvcr_transfer_bytes"] == {("deliver",): 64}


def test_bad_metric_types_and_labels_fail_explicitly(api):
    add_rows = direct_module("metrics").add_rows

    stats = api.stats.OffloadingConnectorStats()
    add_rows(stats, [["count", "counter", [], 1]])
    with pytest.raises(ValueError, match="conflicting"):
        add_rows(stats, [["count", "gauge", [], 2]])
    with pytest.raises(ValueError, match="labels"):
        add_rows(stats, [["other", "counter", [3], 2]])


@pytest.mark.parametrize("status", ["HIT", "FETCHABLE"])
def test_load_failure_quarantines_cached_hits_for_only_failed_request(
    api, manager, status
):
    m, c = manager
    failed = api.base.ReqContext("failed", kv_hints={"advisory": "unchanged"})
    other = api.base.ReqContext("other")
    m.on_new_request(failed)
    for req in (failed, other):
        m.lookup(b"same", req)
    tick(api, m)
    for query in [r for r in c.sent if r.kind == "query"]:
        c.answer(query, statuses=[[status, "REMOTE_G2"]])
    tick(api, m)
    assert m.lookup(b"same", failed) is api.base.LookupResult.HIT
    m.lookup(b"queued-before-failure", failed)
    m.mark_load_failure("failed")
    assert m.lookup(b"same", failed) is api.base.LookupResult.MISS
    assert m.lookup(b"never-queried", failed) is api.base.LookupResult.MISS
    assert m.lookup(b"same", other) is api.base.LookupResult.HIT
    tick(api, m)
    assert not any(
        r.kind == "query" and b"queued-before-failure" in r.keys for r in c.sent
    )
    assert failed.kv_hints == {"advisory": "unchanged"}
    assert not any(r.kind == "finish" for r in c.sent)
    assert list(m.take_events()) == []


def test_load_failure_ignores_inflight_late_hit_and_cleans_up_at_finish(api, manager):
    m, c = manager
    req = api.base.ReqContext("reused")
    m.lookup(b"a", req)
    tick(api, m)
    late = next(r for r in c.sent if r.kind == "query")
    m.mark_load_failure(req.req_id)
    c.answer(late, statuses=[["HIT", "REMOTE_G2"]])
    tick(api, m)
    assert m.lookup(b"a", req) is api.base.LookupResult.MISS
    assert not m._requests[req.req_id].statuses
    m.on_request_finished(req)
    fresh = api.base.ReqContext(req.req_id)
    assert m.lookup(b"a", fresh) is api.base.LookupResult.RETRY
    tick(api, m)
    query = [r for r in c.sent if r.kind == "query"][-1]
    c.answer(query, statuses=[["HIT", "REMOTE_G2"]])
    tick(api, m)
    assert m.lookup(b"a", fresh) is api.base.LookupResult.HIT


def test_receive_failure_counter_once_per_request_generation(api, manager):
    RECEIVE_FAILURES = direct_module("metrics").RECEIVE_FAILURES
    metric_definitions = direct_module("metrics").metric_definitions

    m, _ = manager
    req = api.base.ReqContext("r")
    m.on_new_request(req)
    m.mark_load_failure("r")
    m.mark_load_failure("r")
    assert m.get_stats().data["data"][RECEIVE_FAILURES] == {(): 1}
    assert m.get_stats() is None
    m.on_request_finished(req)
    m.mark_load_failure("r")  # A late notification must not create a generation.
    assert m.get_stats() is None
    m.on_new_request(api.base.ReqContext("r"))
    m.mark_load_failure("r")
    assert m.get_stats().data["data"][RECEIVE_FAILURES] == {(): 1}
    definition = metric_definitions({"enable_telemetry": True})[RECEIVE_FAILURES]
    assert isinstance(definition, api.base.OffloadingCounterMetadata)
    assert definition.labelnames == ()


def test_load_failure_with_telemetry_disabled(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    m = DirectGuardManager(
        dataclasses.replace(
            config, extra_config={**config.extra_config, "enable_telemetry": False}
        ),
        client=FakeClient(),
    )
    req = api.base.ReqContext("r")
    m.on_new_request(req)
    m.mark_load_failure("r")
    assert m.lookup(b"k", req) is api.base.LookupResult.MISS
    assert m.get_stats() is None


@pytest.mark.parametrize(
    "tier,expected",
    [
        ("LOCAL_G2", "HOST"),
        ("REMOTE_G2", "P2P"),
        ("G3", "DISK"),
        (None, "EXTERNAL_UNSPECIFIED"),
        ("FW_G1", "EXTERNAL_UNSPECIFIED"),
        ("FW_G2", "EXTERNAL_UNSPECIFIED"),
    ],
)
@pytest.mark.parametrize("status", ["HIT", "FETCHABLE"])
def test_load_source_uses_cached_query_tier_without_changing_hit_logic(
    api, manager, tier, expected, status
):
    m, c = manager
    req = api.base.ReqContext("r")
    assert m.lookup(b"a", req) is api.base.LookupResult.RETRY
    tick(api, m)
    query = next(r for r in c.sent if r.kind == "query")
    c.answer(query, statuses=[[status, tier]])
    tick(api, m)
    sent_before = len(c.sent)
    assert m.get_load_source(b"a", req) is getattr(api.sources.CacheHitSource, expected)
    assert m.lookup(b"a", req) is api.base.LookupResult.HIT
    assert len(c.sent) == sent_before and list(m.take_events()) == []


def test_load_source_unknown_or_failed_request_is_unspecified_without_query(
    api, manager
):
    m, c = manager
    req = api.base.ReqContext("r")
    unknown = api.sources.CacheHitSource.EXTERNAL_UNSPECIFIED
    assert m.get_load_source(b"a", req) is unknown
    assert not c.sent and not m.has_pending_work()
    m.lookup(b"a", req)
    tick(api, m)
    query = next(r for r in c.sent if r.kind == "query")
    c.answer(query, statuses=[["HIT", "REMOTE_G2"]])
    tick(api, m)
    assert m.get_load_source(b"a", req) is api.sources.CacheHitSource.P2P
    m.mark_load_failure(req.req_id)
    assert m.get_load_source(b"a", req) is unknown


@pytest.mark.parametrize("operation", ["lookup", "touch"])
def test_cold_key_bookkeeping_does_not_rescan_prior_keys(api, manager, operation):
    class CountedDict(dict):
        scanned = 0

        def keys(self):
            self.scanned += len(self)
            return super().keys()

    m, _ = manager
    req = api.base.ReqContext("large")
    state = m._state(req.req_id)
    state.queued = CountedDict()
    state.touched = CountedDict()
    count = 2048
    for i in range(count):
        key = i.to_bytes(8, "big")
        if operation == "lookup":
            assert m.lookup(key, req) is api.base.LookupResult.RETRY
        else:
            m.touch([key], req)
    assert state.queued.scanned + state.touched.scanned <= 2 * count


def test_key_bound_tracks_overlapping_touch_query_and_withdrawal(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    config = dataclasses.replace(
        config, extra_config={**config.extra_config, "metadata_max_keys_per_request": 1}
    )
    c = FakeClient()
    m = DirectGuardManager(config, client=c)
    req = api.base.ReqContext("r")
    m.touch([b"a", b"a"], req)
    m.lookup(b"a", req)
    tick(api, m)
    with pytest.raises(BufferError):
        m.touch([b"b"], req)
    query = next(r for r in c.sent if r.kind == "query")
    c.answer(query, statuses=[["HIT", "LOCAL_G2"]])
    tick(api, m)
    with pytest.raises(BufferError):
        m.lookup(b"b", req)
    poll = next(r for r in c.sent if r.kind == "poll")
    c.answer(poll, events=[{"keys": [b"a"], "removed": True, "tier": "LOCAL_G2"}])
    tick(api, m)
    assert m.lookup(b"b", req) is api.base.LookupResult.RETRY


def test_stale_epoch_reply_releases_key_capacity(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    config = dataclasses.replace(
        config, extra_config={**config.extra_config, "metadata_max_keys_per_request": 1}
    )
    c = FakeClient()
    m = DirectGuardManager(config, client=c)
    req = api.base.ReqContext("r")
    m.lookup(b"a", req)
    tick(api, m)
    query = next(r for r in c.sent if r.kind == "query")
    poll = next(r for r in c.sent if r.kind == "poll")
    c.answer(poll, events=[{"keys": [b"a"], "removed": True, "tier": "LOCAL_G2"}])
    tick(api, m)
    with pytest.raises(BufferError):
        m.lookup(b"b", req)
    c.answer(query, statuses=[["HIT", "LOCAL_G2"]])
    tick(api, m)
    assert m.lookup(b"b", req) is api.base.LookupResult.RETRY


def test_touch_bound_is_atomic_and_releases_sent_keys(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    config = dataclasses.replace(
        config, extra_config={**config.extra_config, "metadata_max_keys_per_request": 2}
    )
    c = FakeClient()
    m = DirectGuardManager(config, client=c)
    req = api.base.ReqContext("r")
    m.touch([b"a"], req)
    with pytest.raises(BufferError):
        m.touch([b"b", b"c"], req)
    m.touch([b"b"], req)
    tick(api, m)
    assert next(r for r in c.sent if r.kind == "touch").keys == [b"a", b"b"]
    m.touch([b"c", b"d"], req)


def test_failed_generation_retains_only_inflight_capacity_then_reuses_id(api, config):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    config = dataclasses.replace(
        config, extra_config={**config.extra_config, "metadata_max_keys_per_request": 1}
    )
    c = FakeClient()
    m = DirectGuardManager(config, client=c)
    req = api.base.ReqContext("r")
    m.lookup(b"a", req)
    tick(api, m)
    old_query = next(r for r in c.sent if r.kind == "query")
    m.mark_load_failure("r")
    with pytest.raises(BufferError):
        m.touch([b"b"], req)
    assert m.lookup(b"b", req) is api.base.LookupResult.MISS
    c.answer(old_query, statuses=[["HIT", "REMOTE_G2"]])
    tick(api, m)
    m.touch([b"b"], req)
    assert m.lookup(b"a", req) is api.base.LookupResult.MISS
    m.on_request_finished(req)
    fresh = api.base.ReqContext("r")
    assert m.lookup(b"c", fresh) is api.base.LookupResult.RETRY


@pytest.mark.parametrize(
    "reply,exception",
    [
        ({"statuses": []}, ValueError),
        ({"statuses": [["INVALID", None]]}, ValueError),
        ({"statuses": [["HIT", "INVALID"]]}, ValueError),
        ({"error": "unavailable"}, RuntimeError),
    ],
)
def test_query_error_releases_finished_inflight_key_capacity(
    api, config, reply, exception
):
    DirectGuardManager = direct_module("manager").DirectGuardManager

    config = dataclasses.replace(
        config, extra_config={**config.extra_config, "metadata_max_keys_per_request": 1}
    )
    c = FakeClient()
    m = DirectGuardManager(config, client=c)
    req = api.base.ReqContext("r")
    m.lookup(b"a", req)
    tick(api, m)
    c.answer(next(r for r in c.sent if r.kind == "query"), **reply)
    with pytest.raises(exception):
        tick(api, m)
    assert m.lookup(b"b", req) is api.base.LookupResult.RETRY
