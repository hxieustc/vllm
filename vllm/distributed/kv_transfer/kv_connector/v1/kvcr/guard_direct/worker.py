# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct GPU endpoints with one owner of KVCR's caller API.

The scheduler's existing job fences retain physical GPU pages. This adapter
acks only completed, non-hazardous jobs; misses have explicit invalid IDs.
"""

import ctypes
import hashlib
import json
import os
import queue
import threading
import time
import traceback
import uuid
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any

import msgspec
import zmq

from .hazards import GPUHazards
from .jobs import DirectResult, IOJob
from .layout import PackedLayout
from .visibility import GPUVisibility

# Failed close cannot let Python free buffers still reachable by native DMA.
_RETAINED_NONQUIESCENT_WORKERS: list["DirectIOActor | DirectGuardWorker"] = []


DIRECT_TRANSPORT = dict(
    UCX_NET_DEVICES="mlx5_0:1",
    UCX_TLS="rc,cuda_copy",
    UCX_MAX_RMA_RAILS="1",
    NIXL_UCX_SGL_ENABLE="1",
    KVCR_NVTX_LEVEL="off",
)


def direct_transport_environment():
    """Read Python's mapping AND the C environment consumed by native UCX."""
    libc = ctypes.CDLL(None)
    libc.getenv.argtypes = [ctypes.c_char_p]
    libc.getenv.restype = ctypes.c_char_p
    native = {}
    for name in DIRECT_TRANSPORT:
        value = libc.getenv(name.encode())
        native[name] = value.decode() if value is not None else None
    return dict(
        python={name: os.environ.get(name) for name in DIRECT_TRANSPORT}, libc=native
    )


def prepare_direct_transport(rank, agent):
    """Direct Guard's exclusive NIC contract; no generic framework env policy."""
    prior = direct_transport_environment()
    os.environ["UCX_NET_DEVICES"] = DIRECT_TRANSPORT["UCX_NET_DEVICES"]
    effective = direct_transport_environment()
    if any(values != DIRECT_TRANSPORT for values in effective.values()):
        raise RuntimeError(
            "Direct Guard transport flags differ from the single-NIC contract"
        )
    return dict(
        schema_version=1,
        rank=rank,
        pid=os.getpid(),
        agent=agent,
        owner_tid=threading.get_native_id(),
        prior=prior,
        effective=effective,
        configured_monotonic_ns=time.monotonic_ns(),
    )


class DirectIOActor(threading.Thread):
    def __init__(
        self,
        config,
        layout,
        *,
        client_factory=None,
        visibility=None,
        device_setup=None,
        endpoint=None,
    ):
        super().__init__(name="kvcr-direct-io", daemon=True)
        self.config = config
        self.layout = layout
        self.incoming: queue.Queue[IOJob] = queue.Queue(maxsize=256)
        self.completed: queue.Queue[DirectResult] = queue.Queue()
        self.ready = threading.Event()
        self.stop_requested = threading.Event()
        self.wakeup = threading.Event()
        self.error = None
        self.quiescent = False
        self.client: Any = None
        self.control = None
        self.client_factory = client_factory
        self.visibility = visibility
        self.device_setup = device_setup
        self.endpoint = endpoint
        self.jobs = OrderedDict()
        self.operation_jobs = {}
        self.inventory = []
        self.pending_stats = {}
        self._replies: deque[list[bytes]] = deque()
        self.finished_requests = OrderedDict()
        self.agent = "KVCR-DIRECT-" + uuid.uuid4().hex
        self.hazards = GPUHazards(self.agent)
        self.forbidden_pin_calls = 0
        self.last_stats = 0.0
        self._load_streak = 0
        self.chunk_limit = int(
            config.extra_config.get("direct_keys_per_operation", 256)
        )
        if not 1 <= self.chunk_limit <= 4096:
            raise ValueError("Invalid direct keys-per-operation bound")

    def _event(self, record, *, binary=False):
        from kvcr import compact_telemetry

        capture = compact_telemetry.writer()
        if capture:
            record = dict(agent=self.agent, **record)
            (capture.journal if binary else capture.emit)(record)

    def _forbidden_pin(self, *args):
        self.forbidden_pin_calls += 1
        self._event(dict(event="forbidden_source_pin", guard_only_sources=True))
        raise RuntimeError("Direct Guard forbids framework source pin callbacks")

    def _resilience(self, error):
        from kvcr.types import TransferError

        if not isinstance(error, TransferError):
            raise error
        self.hazards.observe(error)
        self._event(
            dict(
                event="direct_gpu_hazard",
                op_handle=error.op_handle,
                state=error.state,
                held_gpu_blocks=len(set().union(*self.hazards.active.values()))
                if self.hazards.active
                else 0,
            )
        )

    def _create_client(self):
        from kvcr import KVCR, KVCRBindings
        from kvcr.config import (
            KVCRBackendConfigs,
            KVCRConfig,
            KVCRGuardConfig,
            RemoteFWDramOptions,
        )
        from kvcr.control_channels import ZmqPeerControlChannel
        from kvcr.types import CacheTier, RegionDescriptor

        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
            OffloadingConnectorStats,
        )
        from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash

        rank = self.config.parallel.data_parallel_rank_local
        if rank is None:
            rank = self.config.parallel.data_parallel_index
        extra = self.config.extra_config
        ports = extra.get("control_ports", list(range(17771, 17779)))
        self.control = ZmqPeerControlChannel(
            "0.0.0.0", ports[rank], extra.get("control_advertise_host", "127.0.0.1")
        )

        class Keys:
            def encode(self, key):
                return bytes(key)

            def decode(self, key):
                return maybe_convert_block_hash(BlockHash(key[:-4]))

        def inventory(event):
            if event.tier is not CacheTier.LOCAL_G2:
                raise RuntimeError("Unexpected direct inventory tier")
            self.inventory.append(
                dict(keys=list(event.keys), removed=event.removed, tier=event.tier.name)
            )

        arguments = (
            KVCRConfig(
                nixl_agent_name=self.agent,
                pool_layouts=[("", self.layout.page_bytes)],
                enable_telemetry=True,
                operation_timeout_ms=30000,
                abandon_timeout_ms=60000,
                nixl_listen_port=0,
                startup_timeout_seconds=900,
            ),
            KVCRBindings(
                request_pin=self._forbidden_pin,
                poll_pin_results=lambda: (),
                release_pin=self._forbidden_pin,
                cancel_pin_request=self._forbidden_pin,
                framework_control=self.control,
                key_adapter=Keys(),
                inventory_sink=inventory,
                stats_factory=OffloadingConnectorStats,
                on_resilience_event=self._resilience,
            ),
            KVCRBackendConfigs(
                framework_regions=[RegionDescriptor(**self.layout.region_kwargs())],
                remote_fw_dram=RemoteFWDramOptions(
                    guard_only_sources=True,
                    guard_only_source_proof=True,
                    opportunistic_query=False,
                    eager_ctrl_connect=False,
                ),
            ),
            KVCRGuardConfig(
                kvcr_service_socket_path=extra["guard_socket_path"],
                guard_index=rank,
                compatibility_digest=extra["compatibility_digest"],
            ),
        )
        self.transport_proof = prepare_direct_transport(rank, self.agent)
        self.transport_proof["kvcr_create_begin_ns"] = time.monotonic_ns()
        client = KVCR(*arguments)
        try:
            self.transport_proof["kvcr_create_end_ns"] = time.monotonic_ns()
            progress = client._core._progress
            self.transport_proof.update(
                nixl_initialized=(
                    progress._ready.is_set()
                    and progress._nixl_agent is not None
                    and progress.nixl_agent_metadata is not None
                ),
                nixl_progress_tid=progress._thread.native_id,
                after_create=direct_transport_environment(),
            )
            if (
                not self.transport_proof["nixl_initialized"]
                or self.transport_proof["after_create"]
                != self.transport_proof["effective"]
            ):
                raise RuntimeError(
                    "Direct Guard transport/native initialization could not be attested"
                )
        except BaseException:
            try:
                client.close()
            except BaseException:
                # Constructor returned native ownership: failed attestation must
                # not free registered GPU memory if cleanup cannot quiesce.
                self.client = client
                _RETAINED_NONQUIESCENT_WORKERS.append(self)
                raise
            raise
        return client

    def _proof(self):
        if not all(
            callable(getattr(self.client, name, None))
            for name in ("query_ready", "query_ready_observations")
        ):
            raise RuntimeError("Direct confirmed READY query APIs are not available")
        core = self.client._core
        rank = self.config.parallel.data_parallel_rank_local
        if rank is None:
            rank = self.config.parallel.data_parallel_index
        options = core._remote_fw_dram._options
        if (
            options.guard_only_sources is not True
            or options.guard_only_source_proof is not True
            or options.opportunistic_query is not False
        ):
            raise RuntimeError("Direct source flags are not active")
        proof = dict(
            rank=rank,
            pid=os.getpid(),
            agent=self.agent,
            transport=self.transport_proof,
            guard_only_sources=True,
            guard_only_source_proof=True,
            remote_query_mode="source_confirmed_ready",
            opportunistic_query=False,
            primary_inventory_published=False,
            primary_host_data_bytes=0,
            guard_pool_bytes=sum(pool[1] for pool in core._local_dram._pools.values()),
            framework_region_mem_types=["VRAM"],
            direct_gpu_guard=True,
            gpu_region=self.layout.region_kwargs(),
            kv_group_tokens=[g.tokens_per_block for g in self.config.groups],
            group_ids=list(self.layout.group_ids),
            tokens_per_hash=self.config.cache.tokens_per_hash,
            router_hint_block_size=self.config.extra_config.get(
                "router_hint_block_size"
            ),
            source_pin_callback_calls=self.forbidden_pin_calls,
            metadata_endpoint=self.endpoint,
            data_paths=["GPU->own Guard", "own Guard->GPU", "hinted remote Guard->GPU"],
            visibility=self.visibility.proof(),
            adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        run = Path(os.environ["RUN_DIR"])
        (run / f"option-a-rank-{rank}.json").write_text(json.dumps(proof) + "\n")
        (run / f"layout-rank-{rank}.json").write_text(
            json.dumps(
                dict(
                    rank=rank,
                    block_size_bytes=self.layout.page_bytes,
                    gpu_block_count=self.layout.count,
                    group_ids=self.layout.group_ids,
                    framework_host_data_bytes=0,
                )
            )
            + "\n"
        )

    def _stats(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_stats < 1:
            return
        self.last_stats = now
        stats = self.client.get_stats()
        if stats is not None:
            for name, values in stats._values.items():
                kind = stats._types[name]
                for labels, value in values.items():
                    key = (name, kind, labels)
                    if kind == "counter":
                        self.pending_stats[key] = self.pending_stats.get(key, 0) + value
                    elif kind == "gauge":
                        self.pending_stats[key] = value
                    elif kind == "histogram":
                        self.pending_stats.setdefault(key, []).extend(value)
                    else:
                        raise ValueError("Unknown native metric type")

    def _flush_replies(self, socket):
        for _ in range(min(16, len(self._replies))):
            try:
                socket.send_multipart(self._replies[0], flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            self._replies.popleft()

    def _metadata(self, socket):
        self._flush_replies(socket)
        for _ in range(16):
            if len(self._replies) >= 1024:
                break
            try:
                identity, payload = socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            message = msgspec.msgpack.decode(payload)
            response = {"id": message["id"]}
            try:
                kind = message["kind"]
                request = message.get("request_id")
                if kind == "query":
                    response["statuses"] = [
                        [s.name, t.name if t is not None else None]
                        for s, t in self.client.query_ready(
                            message["keys"], request, timeout_ms=5000
                        )
                    ]
                    response["observations"] = self.client.query_ready_observations(
                        message["keys"], request
                    )
                elif kind == "hint":
                    self.client.submit_hint(message["hint"], request)
                elif kind == "finish":
                    self.finished_requests[request] = None
                    while len(self.finished_requests) > 4096:
                        self.finished_requests.popitem(last=False)
                    if not any(j.request_id == request for j in self.jobs.values()):
                        self.client.discard_hint(request)
                elif kind == "touch":
                    self.client.align_sequence(message["keys"], use_current_time=True)
                elif kind != "poll":
                    raise ValueError("Unknown direct metadata operation")
            except Exception as error:
                response["error"] = f"{type(error).__name__}: {error}"
            self._stats()
            response.update(
                events=self.inventory,
                stats=[
                    [name, kind, list(labels), value]
                    for (name, kind, labels), value in self.pending_stats.items()
                ],
            )
            self.inventory = []
            self.pending_stats = {}
            self._replies.append([identity, msgspec.msgpack.encode(response)])
        self._flush_replies(socket)

    def _ingest(self):
        for _ in range(16):
            try:
                job = self.incoming.get_nowait()
            except queue.Empty:
                break
            if job.job_id in self.jobs:
                raise ValueError("Duplicate live direct GPU job")
            self.jobs[job.job_id] = job
            self._event(
                dict(
                    event="direct_gpu_submit",
                    job_id=job.job_id,
                    kind=job.kind,
                    request_id=job.request_id,
                    keys=list(job.mapping),
                    gpu_indices=list(job.mapping.values()),
                    framework_host_data_bytes=0,
                ),
                binary=True,
            )

    def _launch(self):
        from kvcr.types import MemoryRef

        # Four GET admissions then one PUT when both kinds are ready.
        # Within a kind the OrderedDict rotation is round-robin across jobs.
        loads = deque(j for j in self.jobs.values() if j.kind == "load" and j.unsent)
        stores = deque(j for j in self.jobs.values() if j.kind == "store" and j.unsent)
        while (loads or stores) and len(self.operation_jobs) < 16:
            if stores and (self._load_streak >= 4 or not loads):
                job = stores.popleft()
                self._load_streak = 0
            else:
                job = loads.popleft()
                self._load_streak += 1
            if job.ready_ns is None:
                if not job.ready_event.query():
                    continue
                job.ready_ns = time.monotonic_ns()
                if job.hint is not None:
                    self.client.submit_hint(job.hint, job.request_id)
            keys = job.unsent[: self.chunk_limit]
            refs = {
                key: [
                    MemoryRef(end_point_name=self.agent, element_index=job.mapping[key])
                ]
                for key in keys
            }
            handle = (
                self.client.deposit(refs, expected_layout=[""])
                if job.kind == "store"
                else self.client.deliver(refs, job.request_id)
            )
            job.add_operation(handle, keys)
            del job.unsent[: len(keys)]
            self.operation_jobs[handle] = job
            self._event(
                dict(
                    event="direct_gpu_api",
                    job_id=job.job_id,
                    kind=job.kind,
                    op_handle=handle,
                    keys=keys,
                    gpu_indices=[job.mapping[k] for k in keys],
                ),
                binary=True,
            )
            self.jobs.move_to_end(job.job_id)

    def _poll(self):
        # KVCR delivers uncertain/quiesced callbacks before returning entries.
        for handle, entries in self.client.poll_completed():
            job = self.operation_jobs.get(handle)
            if job is None:
                raise RuntimeError("Unowned KVCR API completion")
            job.complete(handle, entries)
            if handle not in job.batches:
                del self.operation_jobs[handle]
        for job_id, job in list(self.jobs.items()):
            if not job.releasable(self.hazards):
                continue
            if job.kind == "load":
                self.visibility.complete_reads()
            result = job.result()
            self._event(
                dict(
                    event="direct_gpu_terminal",
                    job_id=job_id,
                    kind=job.kind,
                    request_id=job.request_id,
                    requested_blocks=len(job.mapping),
                    successful_blocks=len(job.successful),
                    failed_blocks=len(job.failed),
                    successful_key_bytes=result.transfer_size,
                    bytes_scope=(
                        "Successful API key payload; "
                        "resident PUT may not issue DMA; "
                        "native samples are separate"
                    ),
                    seconds=result.transfer_time,
                    submit_ns=job.submitted_ns,
                    cuda_ready_ns=job.ready_ns,
                    first_api_ns=job.first_api_ns,
                    last_api_ns=job.last_api_ns,
                    cleanup_complete_ns=time.monotonic_ns(),
                    gpu_hazards=0,
                    framework_host_data_bytes=0,
                )
            )
            self._event(
                dict(
                    event="direct_gpu_result",
                    job_id=job_id,
                    kind=job.kind,
                    request_id=job.request_id,
                    keys=list(job.mapping),
                    gpu_indices=list(job.mapping.values()),
                    successful_indices=[
                        i for i, k in enumerate(job.mapping) if k in job.successful
                    ],
                    failed_indices=[
                        i for i, k in enumerate(job.mapping) if k in job.failed
                    ],
                ),
                binary=True,
            )
            self.completed.put(result)
            del self.jobs[job_id]
            if job.request_id in self.finished_requests and not any(
                j.request_id == job.request_id for j in self.jobs.values()
            ):
                self.client.discard_hint(job.request_id)

    def run(self):
        context = socket = None
        try:
            if self.device_setup is not None:
                self.device_setup()
            else:
                import torch

                torch.accelerator.set_device_index(self.layout.device_index)
            if self.visibility is None:
                self.visibility = GPUVisibility.for_device(self.layout.device_index)
            self.client = (
                self.client_factory(self)
                if self.client_factory
                else self._create_client()
            )
            if self.endpoint is None:
                from .protocol import metadata_endpoint

                self.endpoint = metadata_endpoint(self.config)
            context = zmq.Context()
            socket = context.socket(zmq.ROUTER)
            socket.linger = 0
            socket.setsockopt(zmq.SNDHWM, 256)
            socket.setsockopt(zmq.RCVHWM, 256)
            socket.bind(self.endpoint)
            if self.client_factory is None:
                self._proof()
            self.ready.set()
            while not (
                self.stop_requested.is_set() and self.incoming.empty() and not self.jobs
            ):
                self._metadata(socket)
                self._ingest()
                self._poll()
                self._launch()
                self._stats()
                self.wakeup.wait(0.0005 if self.jobs else 0.002)
                self.wakeup.clear()
            self._stats(force=True)
        except BaseException:
            self.error = traceback.format_exc()
            self.ready.set()
        finally:
            if self.client is not None:
                try:
                    self.client.close()
                    self.quiescent = True
                except BaseException:
                    self.error = (
                        (self.error or "") + "\nClose failed: " + traceback.format_exc()
                    )
            if self.control is not None and self.quiescent:
                self.control.close()
            if socket is not None:
                socket.close()
            if context is not None:
                context.term()


class DirectGuardWorker:
    def __init__(self, config, kv_caches):
        self.layout = PackedLayout(config, kv_caches)
        self.config = config
        self.actor = DirectIOActor(config, self.layout)
        self._active = set()
        self._finished = OrderedDict()
        self.actor.start()
        if not self.actor.ready.wait(950):
            raise RuntimeError("Direct Guard initialization did not finish")
        self._check()

    def _check(self):
        if self.actor.error:
            raise RuntimeError("Direct Guard I/O actor failed:\n" + self.actor.error)
        if not self.actor.is_alive() and not self.actor.stop_requested.is_set():
            raise RuntimeError("Direct Guard I/O actor stopped unexpectedly")

    def _submit(self, job_id, kind, spec, gpu_spec):
        import torch

        self._check()
        if job_id in self._active or job_id in self._finished:
            raise ValueError("Duplicate direct GPU job ID")
        mapping = self.layout.map_indices(spec.keys, gpu_spec)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.layout.device_index))
        job = IOJob(
            job_id,
            kind,
            spec.request_id,
            mapping,
            ready,
            getattr(spec, "hint", None),
            self.layout.page_bytes,
        )
        self.actor.incoming.put(job, timeout=5)
        self._active.add(job_id)
        self.actor.wakeup.set()
        return True

    def submit_store(self, job_id, src_spec, dst_spec):
        return self._submit(job_id, "store", dst_spec, src_spec)

    def submit_load(self, job_id, src_spec, dst_spec):
        return self._submit(job_id, "load", src_spec, dst_spec)

    def _drain(self):
        self._check()
        while True:
            try:
                result = self.actor.completed.get_nowait()
            except queue.Empty:
                break
            self._active.remove(result.job_id)
            self._finished[result.job_id] = result

    def get_finished(self):
        self._drain()
        results = list(self._finished.values())
        self._finished.clear()
        return results

    def wait(self, job_ids):
        deadline = time.monotonic() + 600
        while self._active & job_ids:
            self._drain()
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "GPU reuse fence still waits for direct KVCR completion/quiescence"
                )
            time.sleep(0.0005)

    def shutdown(self):
        self.actor.stop_requested.set()
        self.actor.wakeup.set()
        self.actor.join(120)
        if self.actor.is_alive() or not self.actor.quiescent:
            _RETAINED_NONQUIESCENT_WORKERS.append(self)
            raise RuntimeError(
                "Direct Guard shutdown retains GPU buffers until native quiescence"
            )
        self._check()
