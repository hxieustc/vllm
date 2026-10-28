# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned offloading stats conversion and constant-cardinality direct metrics."""

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
)

METADATA_QUERIES = "vllm:kvcr_direct_metadata_queries"
METADATA_KEYS = "vllm:kvcr_direct_metadata_query_keys"
RECEIVE_FAILURES = "vllm:kvcr_direct_receive_failures"


def metric_definitions(extra_config):
    if not extra_config.get("enable_telemetry", False):
        return {}
    # Reuse _kvcr_metric_definitions from the pinned adapter verbatim in value.
    # Do not import that module: its native MemDescriptor API was removed in
    # the custom KVCR snapshot. CPU tests compare against the actual helper.
    return {
        "vllm:kvcr_duration_seconds": OffloadingHistogramMetadata(
            documentation="KVCR operation and stage duration in seconds.",
            labelnames=("scope", "result"),
        ),
        "vllm:kvcr_transfer_bytes": OffloadingCounterMetadata(
            documentation="Bytes transferred by KVCR operations.",
            labelnames=("operation",),
        ),
        "vllm:kvcr_transfer_blocks": OffloadingCounterMetadata(
            documentation="KV blocks transferred by successful KVCR operations.",
            labelnames=("operation",),
        ),
        "vllm:kvcr_state": OffloadingGaugeMetadata(
            documentation="Current KVCR metadata and operation counts.",
            labelnames=("resource",),
        ),
        "vllm:kvcr_guard_source_events": OffloadingCounterMetadata(
            documentation="Guard-only source residency, claim and callback events.",
            labelnames=("event",),
        ),
        METADATA_QUERIES: OffloadingCounterMetadata(
            documentation="Batched scheduler metadata queries submitted to the worker."
        ),
        METADATA_KEYS: OffloadingCounterMetadata(
            documentation="Keys submitted in batched scheduler metadata queries."
        ),
        RECEIVE_FAILURES: OffloadingCounterMetadata(
            documentation=(
                "Request generations quarantined after direct KV receive failure."
            )
        ),
    }


def add_rows(stats: OffloadingConnectorStats, rows) -> None:
    """Reconstruct typed rows; histogram rows may contain a sample or samples."""
    for name, kind, labels, value in rows:
        if not isinstance(name, str) or not isinstance(labels, (list, tuple)):
            raise ValueError("invalid metadata metric row")
        if not all(isinstance(label, str) for label in labels):
            raise ValueError("metric labels must be strings")
        name = name if name.startswith("vllm:") else f"vllm:{name}"
        labelvalues = tuple(labels)
        previous = stats.data["types"].get(name)
        if previous is not None and previous != kind:
            raise ValueError(f"conflicting metric types for {name}")
        values = value if kind == "histogram" and isinstance(value, list) else [value]
        if not all(type(sample) in (int, float) for sample in values):
            raise ValueError("metric values must be numeric")
        if kind == "counter":
            stats.increase_counter(name, value, labelvalues)
        elif kind == "gauge":
            stats.set_gauge(name, value, labelvalues)
        elif kind == "histogram":
            for sample in values:
                stats.observe_histogram(name, sample, labelvalues)
        else:
            raise ValueError(f"unknown metadata metric type: {kind}")
