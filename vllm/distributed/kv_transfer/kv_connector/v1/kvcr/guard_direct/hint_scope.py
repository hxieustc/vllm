# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, request-local authorization below explicit canonical router hashes.

Positions are supplied by the pinned scheduler, never by the router envelope.
This only expands advisory lookup scope; Guard still decides physical READY.
"""

import hashlib
import math
from collections import Counter, defaultdict
from copy import deepcopy

import msgspec


def validate_hint_block_size(config, *, required=False):
    value = config.extra_config.get("router_hint_block_size")
    if value is None and not required:
        return None
    if type(value) is not int or value != 256:
        raise ValueError("direct hints require explicit router_hint_block_size=256")
    if (
        config.cache.blocks_per_chunk != 1
        or not config.groups
        or any(
            type(g.tokens_per_block) is not int or g.tokens_per_block <= 0
            for g in config.groups
        )
        or math.lcm(*(g.tokens_per_block for g in config.groups)) != value
    ):
        raise ValueError(
            "router_hint_block_size must equal group LCM "
            "for this restricted BPC1 adapter"
        )
    unit = config.cache.tokens_per_hash
    if type(unit) is not int or unit <= 0 or value % unit:
        raise ValueError("router_hint_block_size must align to the request hash unit")
    for group in config.groups:
        span = group.tokens_per_block * config.cache.blocks_per_chunk
        if span <= 0 or span % unit or value % span:
            raise ValueError("router_hint_block_size must contain whole group chunks")
    if not any(
        g.tokens_per_block * config.cache.blocks_per_chunk == value
        for g in config.groups
    ):
        raise ValueError("router_hint_block_size requires an actual canonical group")
    return value


def external_hash(raw):
    # Pinned maybe_convert_block_hash with VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=1.
    return int.from_bytes(raw, "big") & ((1 << 64) - 1)


class HintScope:
    @classmethod
    def create(cls, config, hint, limit):
        for index, action in enumerate(hint.get("actions", ())):
            if isinstance(action, dict) and action.get("action_type") == "kv.fetch":
                return cls(config, hint, index, limit)
        return None  # Other advisory envelope types retain their old behavior.

    def __init__(self, config, hint, action_index, limit):
        self.block_size = validate_hint_block_size(config, required=True)
        self.spans = {
            g.group_id: g.tokens_per_block * config.cache.blocks_per_chunk
            for g in config.groups
        }
        payload = hint["actions"][action_index].get("payload", {})
        hashes = payload.get("block_hashes")
        if (
            not isinstance(payload.get("source_control_endpoint"), str)
            or not payload["source_control_endpoint"]
            or not isinstance(hashes, list)
            or not hashes
            or any(type(h) is not int or not 0 <= h < 1 << 64 for h in hashes)
        ):
            raise ValueError("invalid kv.fetch scope")
        if len(hashes) > limit:
            raise BufferError("direct hint parent bound exceeded")
        self.original = deepcopy(hint)
        self.parents = set(hashes)
        self.action_index, self.limit = action_index, limit
        self.candidates = {}  # key -> (end, span, parent_end, raw_hash) or None
        self.parent_hashes = {}  # canonical end -> actual full hash; None = conflict
        self.children = defaultdict(set)
        self.dirty = set()
        self.authorized = {}
        self.hash_refs: Counter[int] = Counter()
        self.normalized = None
        self.proof = None
        self._render_dirty = True
        self.original_digest = hashlib.sha256(
            msgspec.msgpack.encode(self.original)
        ).hexdigest()

    def observe(self, key, end):
        if key not in self.candidates and len(self.candidates) >= self.limit:
            raise BufferError("direct hint candidate bound exceeded")
        group = (
            int.from_bytes(key[-4:], "big")
            if isinstance(key, bytes) and len(key) > 4
            else None
        )
        span = self.spans.get(group)
        info = None
        if span and type(end) is int and end > 0 and end % span == 0:
            parent_end = -(-end // self.block_size) * self.block_size
            info = (end, span, parent_end, key[:-4])
        if key in self.candidates and self.candidates[key] == info:
            return
        if key in self.candidates and self.candidates[key] is not None:
            # A key's request position is immutable. Contradiction fails closed.
            old = self.candidates[key]
            self.parent_hashes[old[2]] = None
            self.dirty.update(self.children[old[2]])
            info = None
        self.candidates[key] = info
        self.dirty.add(key)
        if info is None:
            return
        end, span, parent_end, raw = info
        self.children[parent_end].add(key)
        if end == parent_end and span == self.block_size:
            if parent_end not in self.parent_hashes:
                if len(self.parent_hashes) >= self.limit:
                    raise BufferError("direct hint position bound exceeded")
                self.parent_hashes[parent_end] = raw
                self.dirty.update(self.children[parent_end])
            elif (
                self.parent_hashes[parent_end] is not None
                and self.parent_hashes[parent_end] != raw
            ):
                self.parent_hashes[parent_end] = None
                self.dirty.update(self.children[parent_end])

    def allows(self, key):
        return key in self.authorized and self._contained(self.candidates.get(key))

    def _contained(self, info):
        if info is None:
            return False
        end, span, parent_end, _ = info
        parent = self.parent_hashes.get(parent_end)
        return (
            end - span >= parent_end - self.block_size
            and parent is not None
            and external_hash(parent) in self.parents
        )

    def release(self, keys):
        for key in keys:
            info = self.candidates.pop(key, None)
            if info is not None:
                self.children[info[2]].discard(key)
                if not self.children[info[2]]:
                    del self.children[info[2]]
            self.dirty.discard(key)
            if key in self.authorized:
                raw = self.authorized.pop(key)
                self.hash_refs[raw] -= 1
                if not self.hash_refs[raw]:
                    del self.hash_refs[raw]
                    self._render_dirty = True

    def flush(self):
        dirty, self.dirty = self.dirty, set()
        changed = set()
        for key in dirty:
            info = self.candidates.get(key)
            allowed = self._contained(info)
            if allowed and info is not None and key not in self.authorized:
                hashed = external_hash(info[3])
                self.authorized[key] = hashed
                if not self.hash_refs[hashed]:
                    self._render_dirty = True
                self.hash_refs[hashed] += 1
                changed.add(key)
            elif not allowed and key in self.authorized:
                hashed = self.authorized.pop(key)
                self.hash_refs[hashed] -= 1
                if not self.hash_refs[hashed]:
                    del self.hash_refs[hashed]
                    self._render_dirty = True
                changed.add(key)
        rendered = self._render_dirty
        if rendered:
            self.normalized = deepcopy(self.original)
            payload = self.normalized["actions"][self.action_index]["payload"]
            payload["block_hashes"] = list(
                dict.fromkeys([*payload["block_hashes"], *self.hash_refs])
            )
            self._render_dirty = False
        if dirty or rendered:
            self.proof = dict(
                router_hint_block_size=self.block_size,
                original_parent_hashes=len(self.parents),
                candidate_keys=len(self.candidates),
                authorized_keys=len(self.authorized),
                denied_keys=len(self.candidates) - len(self.authorized),
                authorized_groups=dict(
                    Counter(int.from_bytes(k[-4:], "big") for k in self.authorized)
                ),
                verified_parent_positions=sum(
                    raw is not None and external_hash(raw) in self.parents
                    for raw in self.parent_hashes.values()
                ),
                original_hint_sha256=self.original_digest,
                normalized_hint_sha256=hashlib.sha256(
                    msgspec.msgpack.encode(self.normalized)
                ).hexdigest(),
                scope=(
                    "Advisory authorization only; no READY or transfer completion claim"
                ),
            )
        return self.normalized, changed, rendered, bool(dirty or rendered)
