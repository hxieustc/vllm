# Direct GPU ↔ KVCR Guard: configuration and reproduction notes

These usage files are published on `harryx/guard-direct-docs`, based on the
code-only `harryx/guard-direct` branch. Production code is in
`vllm/distributed/kv_transfer/kv_connector/v1/kvcr/guard_direct/`, mirroring
`mooncake/store`. Both the connector and offloading spec are registered lazily;
no additional adapter `PYTHONPATH` or custom module-path settings are needed.

## Source pins

- vLLM base: `43b4aaea3e40e20ef53ce2bfecfa1b72060bf2b5` plus the rewritten
  `hxieustc/vllm:harryx/guard-direct` commits.
- KVCR: `ai-dynamo/kvcr:harryx/guard-direct`, commit
  `0cc2cc94b987cadce6c65eb000b61f425ab1ee7a`.
- Tested serving dependency pins: Dynamo `1.6.0.dev20261014`, NIXL 1.5.0,
  Python 3.12, eight B200 workers, DeepSeek-V4-Pro DSpark snapshot
  `7c09739fd136abfb70a49ec334157f65f45b52cd`.

The pinned vLLM base already includes the hint/context, ownership-event and
failed-receive interfaces used here. This branch does not vendor Dynamo or
KVCR. It retains r18 behavior with repository formatting and the documented
`torch.accelerator.set_device_index()` device-selection adaptation.

## Payload paths and serving setup

- PUT: GPU → own Guard DRAM via `deposit()`.
- Local GET: own Guard DRAM → GPU via `deliver()`.
- Remote GET: hinted peer Guard DRAM → target GPU through a NIXL WRITE.

There is no framework CPU payload cache, target Guard relay, centralized metadata
server or source-side framework pin fallback. A fresh READY-subset export claim
fences the source; vLLM GPU reservations and native hazard tracking fence the
destination. Failed hybrid loads use request-level recomputation.

Use the sibling `kv-transfer.json` with `--kv-transfer-config`. Keep its
`secondary_tiers` entry: Dynamo uses it for endpoint/capability discovery even
though the custom spec instantiates no CPU tier manager. The configuration is
for eight DP engines in one pod, TP1/PP1/CP1, packed GPU pages and 256-token
canonical hint spans. Loopback advertisements require this shared-pod topology.

Each Guard owns 124 GiB, 992 GiB total. Framework CPU payload capacity is zero.
Allow additional RAM for journals, native state and model loading. Start the
external memory service in its own supervised process:

```bash
.venv/bin/python -m kvcr.kvcr_service \
  --socket-path /run/kvcr/memory.sock --pool-dir /dev/shm/kvcr \
  --guard-count 8 --pool-sizes-gb 124 \
  --compatibility-digest dsv4-dspark-fp8-mxfp4-bs256-direct-guard-v1
```

The serving scheduler and worker processes need:

```bash
export UCX_TLS=rc,cuda_copy UCX_NET_DEVICES=mlx5_0:1 UCX_MAX_RMA_RAILS=1
export NIXL_UCX_SGL_ENABLE=1 KVCR_NVTX_LEVEL=off
export VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=1
export KVCR_DIRECT_META_DIR=/dev/shm/kvcr-direct-ipc
export RUN_DIR=/workspace/direct-guard-run
mkdir -p "$KVCR_DIRECT_META_DIR" "$RUN_DIR" /run/kvcr /dev/shm/kvcr
```

R18 uses KV routing with soft session affinity, affinity TTL 3,600 seconds and
active-decode weight 64. These Dynamo policy settings are separate from the vLLM
adapter. The effective actor operation/abandon timeouts remain 30/60 seconds;
the historical JSON's 5/10-second fields are not wired into its constructor.

## Validation

From the vLLM checkout, run the CPU contracts in an environment with pytest,
NumPy, msgspec and pyzmq:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  --confcutdir=tests/v1/kv_connector/unit/kvcr_guard_direct \
  tests/v1/kv_connector/unit/kvcr_guard_direct -q
```

Without the paired KVCR source checkout, two cross-repository checks are skipped;
worker unit tests use an explicit MemoryRef fake. To include the real KVCR
core/actor/manager bridge and emitted-metric checks, install KVCR's import
dependencies and set `KVCR_SOURCE_DIR` to the pinned KVCR checkout before running
the same command.

CPU tests do not establish GPU or RDMA correctness. Preserve live byte-equality,
noncontiguous-page, READY-subset, missing-key sentinel, source-caller-paused and
native-quiescence canaries before validating a new serving stack.

The [design document](../../../docs/design/kvcr_guard_direct.md) describes the
architecture, API boundaries, operation sequences and optimization opportunities.

Install the in-tree vLLM branch before launch. Historical r18 harness scripts
that require `kvcr_direct.connector` or `kvcr_direct.spec` installation-proof
strings must be updated to the fully qualified in-tree class paths; use this
registered-name configuration rather than their old out-of-tree renderer.
