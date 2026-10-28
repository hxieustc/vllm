# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Restricted startup geometry contract, using the installed vLLM/Dynamo helpers.

Worker proofs check the startup config clone's fallback. Scheduler cache_config
may already be rewritten by EngineCore; its minimum is NOT a frontend fallback.
Metadata here is a TP1/CP1 spec projection, not an observed manager utility reply.
"""

import hashlib
import inspect
import json
import os
from pathlib import Path


def _source(function):
    source = inspect.getsourcefile(function)
    if source is None:
        raise ValueError("Router geometry helper has no inspectable source file")
    path = Path(source).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def validate_router_geometry(vllm_config, kv_cache_config, *, require_fallback):
    from dynamo.vllm.cache_info import (
        MAIN_ATTENTION_KV_CACHE_KINDS,
        get_configured_kv_event_block_size,
        select_main_attention_block_size,
    )

    from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
    from vllm.v1.kv_cache_interface import get_kv_cache_spec_kind

    pc = vllm_config.parallel_config
    if any(
        getattr(pc, name) != 1
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "world_size",
            "decode_context_parallel_size",
            "prefill_context_parallel_size",
        )
    ):
        raise ValueError("Router geometry proof requires TP1/PP1/CP1")
    extra = vllm_config.kv_transfer_config.kv_connector_extra_config
    expected = extra.get("router_hint_block_size")
    if (
        type(expected) is not int
        or expected != 256
        or extra.get("blocks_per_chunk", 1) != 1
    ):
        raise ValueError(
            "Router geometry requires explicit router_hint_block_size=256 and BPC1"
        )
    metadata = []
    for index, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        if type(spec.block_size) is not int or spec.block_size <= 0:
            raise ValueError("Invalid actual cache group block size")
        metadata.append(
            dict(
                group_idx=index,
                kind=get_kv_cache_spec_kind(spec).value,
                block_size=spec.block_size,
                sliding_window=getattr(spec, "sliding_window", None),
            )
        )
    fallback = vllm_config.cache_config.block_size
    configured = get_configured_kv_event_block_size(vllm_config)
    selected = select_main_attention_block_size(metadata, fallback)
    first_main = next(
        (row for row in metadata if row["kind"] in MAIN_ATTENTION_KV_CACHE_KINDS), None
    )
    if first_main is None or selected != expected:
        raise ValueError(
            "First actual main-attention kind does not "
            "select the router hint block size"
        )
    override = (vllm_config.additional_config or {}).get("dynamo_kv_event_block_size")
    if override is not None and (type(override) is not int or override != expected):
        raise ValueError(
            "Explicit Dynamo KV event block-size override disagrees with router hints"
        )
    if require_fallback and (fallback != expected or configured != expected):
        raise ValueError(
            "Worker startup fallback/configured Dynamo block size "
            "disagrees with router hints"
        )
    scheduler_size, hash_size = resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )
    if (
        scheduler_size != expected
        or type(hash_size) is not int
        or hash_size <= 0
        or expected % hash_size
    ):
        raise ValueError(
            "Actual scheduler/hash resolver disagrees with restricted router geometry"
        )
    sources = {
        name: _source(fn)
        for name, fn in (
            ("get_kv_cache_spec_kind", get_kv_cache_spec_kind),
            ("resolve_kv_cache_block_sizes", resolve_kv_cache_block_sizes),
            ("get_configured_kv_event_block_size", get_configured_kv_event_block_size),
            ("select_main_attention_block_size", select_main_attention_block_size),
        )
    }
    v1 = Path(sources["get_kv_cache_spec_kind"]["path"]).parent
    for name, relative in (
        ("engine_group_metadata", "engine/core.py"),
        ("single_type_manager_block_size", "core/single_type_kv_cache_manager.py"),
    ):
        path = v1 / relative
        sources[name] = dict(
            path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
    return dict(
        verified=True,
        group_metadata=metadata,
        first_main_group=first_main["group_idx"],
        router_hint_block_size=expected,
        fallback_block_size_here=fallback,
        configured_block_size_here=configured,
        metadata_selected_block_size=selected,
        explicit_dynamo_override=override,
        scheduler_block_size=scheduler_size,
        tokens_per_hash=hash_size,
        frontend_fallback_verified=bool(require_fallback),
        frontend_fallback_scope=(
            "worker startup config clone; not a frontend RPC observation"
        ),
        metadata_scope=(
            "TP1/CP1 spec projection of pinned manager block_size; not a utility reply"
        ),
        manager_metadata_observed=False,
        sources=sources,
    )


def write_router_geometry_proof(proof, vllm_config, *, worker):
    pc = vllm_config.parallel_config
    rank = pc.data_parallel_rank_local
    if rank is None:
        rank = pc.data_parallel_index
    role = "worker" if worker else "scheduler"
    value = dict(proof, rank=rank, pid=os.getpid(), role=role)
    directory = Path(os.environ["RUN_DIR"])
    path = directory / f"router-geometry-{role}-{rank}.json"
    temporary = path.with_suffix(f".json.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)
    return value
