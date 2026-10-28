# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local metadata wire contract; this channel never carries KV payloads."""

import hashlib
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import msgspec
import zmq

from vllm.v1.kv_offload.base import LoadStoreSpec


@dataclass
class DirectSpec(LoadStoreSpec):
    keys: tuple[bytes, ...]
    request_id: str
    hint: dict | None = None


class Request(msgspec.Struct):
    id: int
    kind: Literal["query", "hint", "finish", "touch", "poll"]
    request_id: str = ""
    keys: list[bytes] = msgspec.field(default_factory=list)
    hint: Any = None


class Response(msgspec.Struct):
    id: int
    statuses: list[list[str | None]] = msgspec.field(default_factory=list)
    events: list[dict[str, Any]] = msgspec.field(default_factory=list)
    stats: list[list[Any]] = msgspec.field(default_factory=list)
    error: str | None = None
    observations: list[dict[str, Any] | None] = msgspec.field(default_factory=list)


def metadata_endpoint(config) -> str:
    directory = os.environ.get("KVCR_DIRECT_META_DIR")
    if not directory or not Path(directory).is_absolute():
        raise ValueError(
            "KVCR_DIRECT_META_DIR must be an explicit absolute IPC directory"
        )
    identity = msgspec.msgpack.encode(
        [config.engine_id, config.parallel.data_parallel_index]
    )
    digest = hashlib.sha256(identity).hexdigest()[:24]
    path = str(Path(directory) / f"kvcr-direct-{digest}.sock")
    if len(os.fsencode(path)) > 100:
        raise ValueError("KVCR_DIRECT_META_DIR produces an overlong IPC socket path")
    return f"ipc://{path}"


class MetadataClient:
    """Scheduler-owned DEALER. All sends/receives are nonblocking and bounded.

    The worker ROUTER receives exactly [identity, msgpack(Request)] and replies
    [identity, msgpack(Response)]. It must acknowledge every request, including
    hint/finish/touch/poll. Context ownership remains with the caller/ZMQ singleton.
    """

    def __init__(self, config, *, context=None, max_pending: int = 1024):
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        self._max_pending = max_pending
        self._next_id = 0
        self._pending: set[int] = set()
        self._outbox: deque[tuple[int, bytes]] = deque()
        self._closed = False
        endpoint = metadata_endpoint(config)
        self.socket = (context or zmq.Context.instance()).socket(zmq.DEALER)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDHWM, max_pending)
        self.socket.setsockopt(zmq.RCVHWM, max_pending)
        self.socket.connect(endpoint)

    def request(self, kind, request_id="", keys=(), hint=None) -> int:
        if self._closed:
            raise RuntimeError("metadata client is closed")
        if len(self._pending) >= self._max_pending:
            raise BufferError("metadata request window is full")
        request_id_number = self._next_id
        request = Request(request_id_number, kind, request_id, list(keys), hint)
        payload = msgspec.msgpack.encode(request)
        self._next_id += 1
        self._pending.add(request_id_number)
        self._outbox.append((request_id_number, payload))
        return request_id_number

    def flush(self) -> None:
        while self._outbox:
            try:
                self.socket.send(self._outbox[0][1], zmq.NOBLOCK)
            except zmq.Again:
                break
            self._outbox.popleft()

    def drain(self) -> list[Response]:
        responses = []
        for _ in range(self._max_pending):
            try:
                payload = self.socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            response = msgspec.msgpack.decode(payload, type=Response)
            if response.id not in self._pending:
                raise ValueError(
                    f"unknown or duplicate metadata response {response.id}"
                )
            self._pending.remove(response.id)
            responses.append(response)
        return responses

    def has_pending_work(self) -> bool:
        return bool(self._pending)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.socket.close(0)
            self._outbox.clear()
            self._pending.clear()
