#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-free installed API check; does not observe a running model/manager."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace as N


def run_smoke():
    import torch

    from vllm.distributed.kv_transfer.kv_connector.v1.kvcr.guard_direct import (
        router_geometry,
    )

    validate_router_geometry = router_geometry.validate_router_geometry
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        UniformTypeKVCacheSpecs,
    )

    cfg = N(
        parallel_config=N(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            world_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        cache_config=N(
            block_size=256, prefix_match_unit=None, enable_prefix_caching=True
        ),
        additional_config={},
        kv_transfer_config=N(
            kv_connector_extra_config={
                "router_hint_block_size": 256,
                "blocks_per_chunk": 1,
            }
        ),
    )

    def group(size, index):
        return KVCacheGroupSpec(
            layer_names=[f"cpu-spec-{index}"],
            kv_cache_spec=FullAttentionSpec(
                block_size=size, num_kv_heads=1, head_size=64, dtype=torch.float16
            ),
        )

    kv = KVCacheConfig(
        num_blocks=1, kv_cache_tensors=[], kv_cache_groups=[group(256, 0)]
    )
    positive = validate_router_geometry(cfg, kv, require_fallback=True)
    if positive["tokens_per_hash"] != 256:
        raise AssertionError("Unexpected single-group hash resolver result")

    def wrapped(group):
        return KVCacheGroupSpec(
            layer_names=group.layer_names,
            kv_cache_spec=UniformTypeKVCacheSpecs(
                block_size=group.kv_cache_spec.block_size,
                kv_cache_specs={
                    name: group.kv_cache_spec for name in group.layer_names
                },
            ),
        )

    kv.kv_cache_groups = [wrapped(group(256, 0))]
    worker_positive = validate_router_geometry(cfg, kv, require_fallback=True)
    if worker_positive["group_metadata"] != positive["group_metadata"]:
        raise AssertionError(
            "Worker uniform wrapper differs from scheduler flat metadata"
        )
    # Alignment remains 256, but the FIRST main kind is 64 and must be rejected.
    kv.kv_cache_groups = [group(64, 0), group(256, 1)]
    try:
        validate_router_geometry(cfg, kv, require_fallback=True)
    except ValueError as exc:
        negative = str(exc)
        if "First actual main-attention kind" not in negative:
            raise
    else:
        raise AssertionError("First-main 64 accepted despite router scope 256")
    kv.kv_cache_groups = [wrapped(group(64, 0)), wrapped(group(256, 1))]
    try:
        validate_router_geometry(cfg, kv, require_fallback=True)
    except ValueError as exc:
        worker_negative = str(exc)
        if "First actual main-attention kind" not in worker_negative:
            raise
    else:
        raise AssertionError("Worker uniform first-main 64 accepted")
    return dict(
        status="passed",
        scope="installed APIs with real CPU spec objects; no model or live manager",
        positive=positive,
        first_main_64_rejected=negative,
        worker_uniform_positive=worker_positive,
        worker_uniform_first_main_64_rejected=worker_negative,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run_smoke()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))
