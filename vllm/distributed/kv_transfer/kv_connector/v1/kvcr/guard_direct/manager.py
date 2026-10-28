# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata-only scheduler manager. The worker/Guard owns storage and READY."""

import logging
import math
from dataclasses import dataclass, field
from itertools import islice
from time import monotonic
from typing import Any

import msgspec

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.v1.kv_offload.base import (
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
    RequestOffloadingContext,
)
from vllm.v1.metrics.cache_hit_source import CacheHitSource

from .hint_scope import HintScope, validate_hint_block_size
from .metrics import METADATA_KEYS, METADATA_QUERIES, RECEIVE_FAILURES, add_rows
from .protocol import DirectSpec, MetadataClient

logger = logging.getLogger(__name__)


@dataclass
class _RequestState:
    statuses: dict[bytes, tuple[str, str | None]] = field(default_factory=dict)
    # A confirmed miss commits this request to local compute for that key.
    misses: set[bytes] = field(default_factory=set)
    queued: dict[bytes, None] = field(default_factory=dict)
    inflight: set[bytes] = field(default_factory=set)
    # Subset of inflight whose authorization changed after submission.
    deferred_requery: set[bytes] = field(default_factory=set)
    touched: dict[bytes, None] = field(default_factory=dict)
    # Exact union of the four live key containers; never a lifetime history.
    known: set[bytes] = field(default_factory=set)
    hint: dict[str, Any] | None = None
    hint_pending: bool = False
    hint_scope: HintScope | None = None
    load_failed: bool = False
    observations_expire_at: float = 0.0
    remote_identity: tuple | None = None
    remote_observed_at: float = -1.0


class DirectGuardManager(OffloadingManager):
    def __init__(self, config, *, client=None, clock=None):
        self.config = config
        extra = config.extra_config
        validate_hint_block_size(config)
        ttl_ms = extra.get("metadata_status_ttl_ms", 5000)
        if type(ttl_ms) is not int or not 1 <= ttl_ms <= 5000:
            raise ValueError("metadata_status_ttl_ms must be an integer in [1, 5000]")
        self._status_ttl = ttl_ms / 1000
        self._clock = clock if clock is not None else monotonic
        self._max_requests = int(extra.get("metadata_max_requests", 4096))
        self._max_keys = int(extra.get("metadata_max_keys_per_request", 65536))
        self._query_batch_keys = int(extra.get("metadata_query_batch_keys", 32768))
        if not 1 <= self._query_batch_keys <= 65536:
            raise ValueError("metadata_query_batch_keys must be in [1, 65536]")
        if min(self._max_requests, self._max_keys) <= 0:
            raise ValueError("metadata bounds must be positive")
        self._client = (
            client
            if client is not None
            else MetadataClient(
                config, max_pending=int(extra.get("metadata_max_pending", 1024))
            )
        )
        self._requests: dict[str, _RequestState] = {}
        self._queries: dict[
            int, tuple[str, _RequestState, tuple[bytes, ...], int, float]
        ] = {}
        self._controls: set[int] = set()
        self._finishes: dict[str, None] = {}
        self._poll_id: int | None = None
        self._epoch = 0
        self._events: list[OffloadingEvent] = []
        self._stats = OffloadingConnectorStats()
        self._telemetry = bool(extra.get("enable_telemetry", False))
        # Inventory events are emitted only after the local Guard owns READY
        # data.  A request without a router fetch hint cannot have a remote
        # source, so querying absent keys only burns scheduler/Guard CPU.
        self._inventory_gate = bool(extra.get("metadata_inventory_gate", False))
        self._local_ready: set[bytes] = set()
        self._closed = False

    def _state(self, request_id: str) -> _RequestState:
        if self._closed:
            raise RuntimeError("direct manager is closed")
        state = self._requests.get(request_id)
        if state is None:
            if len(self._requests) >= self._max_requests:
                raise BufferError("direct metadata request bound exceeded")
            self._requests[request_id] = state = _RequestState()
        return state

    @staticmethod
    def _forget_unused(state, keys):
        for key in keys:
            if (
                key not in state.statuses
                and key not in state.misses
                and key not in state.queued
                and key not in state.inflight
                and key not in state.touched
            ):
                state.known.discard(key)
                if state.hint_scope is not None:
                    state.hint_scope.release((key,))

    def _ensure_hint(self, state, req_context):
        if state.hint is None and req_context.kv_hints is not None:
            hint = msgspec.to_builtins(req_context.kv_hints)
            scope = HintScope.create(self.config, hint, self._max_keys)
            state.hint, state.hint_scope = hint, scope
            state.hint_pending = True

    def _expire_observations(self, state):
        # One conservative deadline for the whole request cache. Adding a
        # response can shorten it, never extend older observations. Reclaim
        # keys only once per expiration, not by scanning on every lookup.
        if state.statuses and self._clock() >= state.observations_expire_at:
            keys = tuple(state.statuses)
            state.statuses.clear()
            state.observations_expire_at = 0.0
            self._forget_unused(state, keys)

    def _remote_observation_deadline(
        self, state, proof, status, expires_at, current_keys
    ):
        # Both clocks are the same target host's monotonic clock. Never use a
        # remote host timestamp or restart a lifetime when IPC replies drain.
        if proof is None:
            return None
        observed, valid_until = (
            proof.get(name)
            for name in ("observed_at_monotonic", "valid_until_monotonic")
        )
        if (
            any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in (observed, valid_until)
            )
            or valid_until < observed
            or observed > self._clock()
            or proof.get("status") != status
        ):
            raise ValueError("invalid confirmed observation clock/status")
        deadline = min(expires_at, valid_until, observed + self._status_ttl)
        if deadline <= self._clock():
            return None
        if status == "MISS" and proof.get("source_incarnation") is None:
            return deadline  # A bounded timeout observation has no peer identity.
        identity = tuple(
            proof.get(name)
            for name in (
                "request_generation",
                "source_endpoint",
                "source_incarnation",
                "source_progress_epoch",
            )
        )
        if (
            any(not isinstance(v, str) or not v for v in identity[:3])
            or type(identity[3]) is not int
            or identity[3] < 0
            or (
                status != "MISS"
                and (
                    type(proof.get("residency_generation")) is not int
                    or proof["residency_generation"] <= 0
                )
            )
        ):
            raise ValueError("invalid confirmed observation identity")
        if state.hint_scope is not None:
            scope = state.hint_scope
            expected = scope.original["actions"][scope.action_index]["payload"][
                "source_control_endpoint"
            ]
            if identity[1] != expected:
                raise ValueError("confirmed observation source differs from hint")
        if state.remote_identity is not None and identity != state.remote_identity:
            if observed <= state.remote_observed_at or (
                identity[1:3] == state.remote_identity[1:3]
                and identity[3] < state.remote_identity[3]
            ):
                return None  # Late observation cannot roll ownership backward.
            for key in tuple(state.misses):
                state.misses.remove(key)
                if key not in state.inflight and key not in current_keys:
                    state.queued[key] = None
            for key, pair in tuple(state.statuses.items()):
                if pair[1] == "REMOTE_G2" or pair[0] == "MISS":
                    del state.statuses[key]
                    if key not in state.inflight and key not in current_keys:
                        state.queued[key] = None
            state.deferred_requery.update(state.inflight)
        state.remote_identity = identity
        state.remote_observed_at = max(state.remote_observed_at, observed)
        return deadline

    def lookup(self, key, req_context):
        state = self._state(req_context.req_id)
        if state.load_failed:
            return LookupResult.MISS
        if key in state.misses:
            return LookupResult.MISS
        self._expire_observations(state)
        if key not in state.known and len(state.known) >= self._max_keys:
            raise BufferError("direct metadata key bound exceeded")
        self._ensure_hint(state, req_context)
        if self._inventory_gate and state.hint is None and key not in self._local_ready:
            return LookupResult.MISS
        if state.hint_scope is not None:
            state.hint_scope.observe(key, req_context.get_offload_key_position(key))
        observed = state.statuses.get(key)
        if observed is not None:
            status, _ = observed
            if status in ("HIT", "FETCHABLE"):
                if (
                    observed[1] == "REMOTE_G2"
                    and state.hint_scope is not None
                    and not state.hint_scope.allows(key)
                ):
                    state.statuses[key] = ("MISS", None)
                    return LookupResult.MISS
                return LookupResult.HIT
            if status == "MISS":
                return LookupResult.MISS
            # FETCHING is transient: never make it a permanent local RETRY.
        if key not in state.queued and key not in state.inflight:
            if key not in state.known and len(state.known) >= self._max_keys:
                raise BufferError("direct metadata key bound exceeded")
            state.known.add(key)
            state.queued[key] = None
        return LookupResult.RETRY

    def get_load_source(self, key, req_context):
        """Attribute cached lookup provenance without acquiring any residency.

        HOST means physical Guard DRAM, not a framework CPU primary cache.
        P2P remains advisory query provenance, not a successful READY claim.
        """
        state = self._requests.get(req_context.req_id)
        if state is None or state.load_failed:
            return CacheHitSource.EXTERNAL_UNSPECIFIED
        self._expire_observations(state)
        status, tier = state.statuses.get(key, (None, None))
        if status not in ("HIT", "FETCHABLE"):
            return CacheHitSource.EXTERNAL_UNSPECIFIED
        if (
            tier == "REMOTE_G2"
            and state.hint_scope is not None
            and not state.hint_scope.allows(key)
        ):
            return CacheHitSource.EXTERNAL_UNSPECIFIED
        sources: dict[str | None, CacheHitSource] = {
            "LOCAL_G2": CacheHitSource.HOST,
            "REMOTE_G2": CacheHitSource.P2P,
            "G3": CacheHitSource.DISK,
        }
        return sources.get(tier, CacheHitSource.EXTERNAL_UNSPECIFIED)

    def mark_load_failure(self, request_id: str) -> None:
        """Recompute this live request without retrying stale remote hints.

        This changes scheduler metadata only. Existing native jobs and GPU
        reservations still complete through the connector's normal fences.
        """
        state = self._requests.get(request_id)
        if state is None or state.load_failed:
            return
        state.load_failed = True
        state.deferred_requery.clear()
        state.statuses.clear()
        state.misses.clear()
        state.observations_expire_at = 0.0
        state.queued.clear()
        state.known.intersection_update(state.inflight | state.touched.keys())
        if self._telemetry:
            self._stats.increase_counter(RECEIVE_FAILURES)

    def on_new_request(self, req_context):
        state = self._state(req_context.req_id)
        self._ensure_hint(state, req_context)
        return RequestOffloadingContext()

    def prepare_load(self, keys, req_context):
        state = self._state(req_context.req_id)
        self._ensure_hint(state, req_context)
        keys = tuple(keys)
        if state.load_failed:
            raise ValueError("cannot load a quarantined request")
        if state.hint_scope is not None:
            for key in keys:
                if not state.hint_scope.allows(key) and state.statuses.get(key) != (
                    "HIT",
                    "LOCAL_G2",
                ):
                    raise ValueError(
                        "load key lacks verified hint scope or local READY"
                    )
        return DirectSpec(keys, req_context.req_id, state.hint)

    def get_hint_scope_attestation(self, request_id):
        state = self._requests.get(request_id)
        return (
            dict(state.hint_scope.proof)
            if state
            and state.hint_scope is not None
            and state.hint_scope.proof is not None
            else None
        )

    def prepare_store(self, keys, req_context):
        # No CPU chunk allocation, no framework eviction, no staging-batch flag.
        keys = list(keys)
        return PrepareStoreOutput(keys, DirectSpec(tuple(keys), req_context.req_id), [])

    def touch(self, keys, req_context):
        state = self._state(req_context.req_id)
        keys = tuple(keys)
        new_keys = set(keys).difference(state.known)
        if len(state.known) + len(new_keys) > self._max_keys:
            raise BufferError("direct metadata key bound exceeded")
        state.known.update(new_keys)
        state.touched.update(dict.fromkeys(keys))

    def on_request_finished(self, req_context):
        # Native job/hint ownership persists in the worker until its jobs drain.
        if (
            len(self._finishes) >= self._max_requests
            and req_context.req_id not in self._finishes
        ):
            raise BufferError("direct finish queue bound exceeded")
        self._requests.pop(req_context.req_id, None)
        self._finishes[req_context.req_id] = None

    def _drain(self):
        for response in self._client.drain():
            self._controls.discard(response.id)
            if response.id == self._poll_id:
                self._poll_id = None
            query = self._queries.pop(response.id, None)
            if query is not None:
                request_id, state, keys, epoch, expires_at = query
                state.inflight.difference_update(keys)
            if query is not None and response.error is None:
                if len(response.statuses) != len(keys):
                    self._forget_unused(state, keys)
                    raise ValueError("query status count does not match requested keys")
                if response.observations and len(response.observations) != len(keys):
                    self._forget_unused(state, keys)
                    raise ValueError(
                        "query observation count does not match requested keys"
                    )
                observations = response.observations or [None] * len(keys)
                current_keys = set(keys)
                for key, pair, proof in zip(
                    keys, response.statuses, observations, strict=True
                ):
                    if len(pair) != 2 or pair[0] not in (
                        "HIT",
                        "FETCHABLE",
                        "FETCHING",
                        "MISS",
                    ):
                        self._forget_unused(state, keys)
                        raise ValueError("invalid query status")
                    status, tier = pair
                    if tier not in (
                        None,
                        "FW_G1",
                        "FW_G2",
                        "LOCAL_G2",
                        "REMOTE_G2",
                        "G3",
                    ):
                        self._forget_unused(state, keys)
                        raise ValueError("invalid query tier")
                    if (
                        self._requests.get(request_id) is state
                        and not state.load_failed
                        and epoch == self._epoch
                        and self._clock() < expires_at
                        and key not in state.deferred_requery
                    ):
                        valid_until = expires_at
                        if (status in ("HIT", "FETCHABLE") and tier == "REMOTE_G2") or (
                            status == "MISS" and proof is not None
                        ):
                            valid_until = self._remote_observation_deadline(
                                state, proof, status, expires_at, current_keys
                            )
                            if valid_until is None:
                                continue
                        if (
                            tier == "REMOTE_G2"
                            and state.hint_scope is not None
                            and not state.hint_scope.allows(key)
                        ):
                            status, tier = "MISS", None
                        if status == "MISS":
                            state.misses.add(key)
                        else:
                            state.observations_expire_at = (
                                min(state.observations_expire_at, valid_until)
                                if state.statuses
                                else valid_until
                            )
                            state.statuses[key] = (status, tier)
                        state.known.add(key)
            if query is not None:
                for key in keys:
                    if key in state.deferred_requery:
                        state.deferred_requery.remove(key)
                        if (
                            self._requests.get(request_id) is state
                            and not state.load_failed
                        ):
                            state.queued[key] = None
                self._forget_unused(state, keys)
            for event in response.events:
                tier = event["tier"]
                if tier not in ("LOCAL_G2", "G3"):
                    raise ValueError(f"unsupported inventory tier: {tier}")
                keys = event["keys"]
                if type(event["removed"]) is not bool or not all(
                    isinstance(k, bytes) for k in keys
                ):
                    raise ValueError("invalid inventory event")
                if event["removed"]:
                    if self._inventory_gate and tier == "LOCAL_G2":
                        self._local_ready.difference_update(keys)
                    self._epoch += 1
                    for state in self._requests.values():
                        for key in keys:
                            state.statuses.pop(key, None)
                        self._forget_unused(state, keys)
                elif self._inventory_gate and tier == "LOCAL_G2":
                    self._local_ready.update(keys)
                if self.config.enable_kv_cache_events:
                    self._events.append(
                        OffloadingEvent(
                            keys=list(keys),
                            medium=Medium.CPU if tier == "LOCAL_G2" else Medium.STORAGE,
                            removed=event["removed"],
                            ownership="kvcr",
                            removal_expected=True,
                        )
                    )
            if self._telemetry:
                add_rows(self._stats, response.stats)
            if response.error is not None:
                raise RuntimeError(f"direct worker metadata error: {response.error}")

    def on_schedule_end(self, context):
        if self._closed:
            return
        for state in self._requests.values():
            self._expire_observations(state)
        self._drain()
        try:
            # Finish old generations before issuing metadata for a reused ID.
            for request_id in tuple(self._finishes):
                self._controls.add(self._client.request("finish", request_id))
                del self._finishes[request_id]
            for request_id, state in self._requests.items():
                if state.hint_scope is not None and not state.load_failed:
                    hint, changed, rendered, attest = state.hint_scope.flush()
                    state.hint = hint
                    state.hint_pending |= rendered
                    for key in changed:
                        if key in state.misses:
                            state.misses.remove(key)
                            if key not in state.inflight:
                                state.queued[key] = None
                        if key in state.inflight:
                            # Wait for the old query to drain; never accept its
                            # negative or positive under the new hint snapshot.
                            state.deferred_requery.add(key)
                        pair = state.statuses.get(key)
                        if pair is not None and (
                            pair[0] == "MISS" or pair[1] == "REMOTE_G2"
                        ):
                            state.statuses.pop(key)
                            if key not in state.inflight:
                                state.queued[key] = None
                    if attest:
                        logger.info(
                            "KVCR_HINT_SCOPE %s",
                            msgspec.json.encode(state.hint_scope.proof).decode(),
                        )
                if state.hint_pending:
                    self._controls.add(
                        self._client.request("hint", request_id, hint=state.hint)
                    )
                    state.hint_pending = False
                if state.queued:
                    keys = tuple(islice(state.queued, self._query_batch_keys))
                    expires_at = self._clock() + self._status_ttl
                    rid = self._client.request("query", request_id, keys)
                    self._queries[rid] = (
                        request_id,
                        state,
                        keys,
                        self._epoch,
                        expires_at,
                    )
                    state.inflight.update(keys)
                    for key in keys:
                        del state.queued[key]
                    if self._telemetry:
                        self._stats.increase_counter(METADATA_QUERIES)
                        self._stats.increase_counter(METADATA_KEYS, len(keys))
                if state.touched:
                    keys = tuple(state.touched)
                    self._controls.add(self._client.request("touch", request_id, keys))
                    state.touched.clear()
                    self._forget_unused(state, keys)
            if self._poll_id is None:
                self._poll_id = self._client.request("poll")
        except BufferError:
            # Keep unsent work in the manager and retry after replies drain.
            pass
        self._client.flush()
        self._drain()

    def take_events(self):
        events, self._events = self._events, []
        return events

    def get_stats(self):
        if not self._telemetry:
            return None
        stats, self._stats = self._stats, OffloadingConnectorStats()
        return None if stats.is_empty() else stats

    def has_pending_work(self):
        return bool(
            not self._closed
            and (
                self._finishes
                or self._queries
                or self._controls
                or any(
                    s.queued or s.touched or s.hint_pending
                    for s in self._requests.values()
                )
            )
        )

    def shutdown(self):
        if not self._closed:
            self._closed = True
            self._client.close()
            self._requests.clear()
            self._queries.clear()
            self._controls.clear()
            self._finishes.clear()
            self._events.clear()
