# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-key API completion is distinct from safe GPU reservation release."""

import time
from dataclasses import dataclass, field


@dataclass
class DirectResult:
    job_id: int
    success: bool
    transfer_size: int
    transfer_time: float
    kind: str
    failed_block_ids: tuple[int, ...] = ()
    failed_keys: tuple[bytes, ...] = ()


@dataclass
class IOJob:
    job_id: int
    kind: str
    request_id: str
    mapping: dict
    ready_event: object
    hint: object
    page_bytes: int
    submitted_ns: int = field(default_factory=time.monotonic_ns)
    ready_ns: int | None = None
    first_api_ns: int | None = None
    last_api_ns: int | None = None
    pending_keys: set = field(init=False)
    batches: dict = field(default_factory=dict)
    successful: set = field(default_factory=set)
    failed: set = field(default_factory=set)
    unsent: list = field(init=False)

    def __post_init__(self):
        self.pending_keys = set(self.mapping)
        self.unsent = list(self.mapping)

    def add_operation(self, handle, keys):
        keys = set(keys)
        if handle in self.batches or not keys <= self.pending_keys:
            raise ValueError("Invalid native batch ownership")
        self.batches[handle] = keys
        if self.first_api_ns is None:
            self.first_api_ns = time.monotonic_ns()

    def complete(self, handle, entries):
        remaining = self.batches.get(handle)
        if remaining is None or not set(entries) <= remaining:
            raise ValueError("Unknown/duplicate native completion key")
        for key, entry in entries.items():
            (self.successful if entry.success else self.failed).add(key)
            self.pending_keys.remove(key)
            remaining.remove(key)
        self.last_api_ns = time.monotonic_ns()
        if not remaining:
            del self.batches[handle]

    def releasable(self, hazards):
        return (
            not self.pending_keys
            and not self.batches
            and not hazards.blocks(self.mapping.values())
        )

    def result(self):
        if self.pending_keys or self.batches:
            raise RuntimeError("GPU job is still in flight")
        return DirectResult(
            self.job_id,
            not self.failed,
            len(self.successful) * self.page_bytes,
            (time.monotonic_ns() - self.submitted_ns) / 1e9,
            self.kind,
            tuple(sorted({self.mapping[k] for k in self.failed}))
            if self.kind == "load"
            else (),
            tuple(k for k in self.mapping if k in self.failed),
        )
