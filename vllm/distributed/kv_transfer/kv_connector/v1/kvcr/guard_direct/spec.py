# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Out-of-tree pinned vLLM spec; no framework CPU data allocation."""

from vllm.v1.kv_offload.base import OffloadingSpec

from .hint_scope import validate_hint_block_size
from .metrics import metric_definitions

GUARD_POOL_BYTES = 124 * 1024**3


class KVCRDirectGuardSpec(OffloadingSpec):
    @classmethod
    def build_metric_definitions(cls, extra_config):
        return metric_definitions(extra_config)

    def __init__(self, config):
        super().__init__(config)
        validate_hint_block_size(config)
        parallel = config.parallel
        if any(
            getattr(parallel, name) != 1
            for name in ("tp_size", "pp_size", "pcp_size", "dcp_size", "world_size")
        ):
            raise ValueError(
                "direct Guard requires TP1/PP1/PCP1/DCP1 and one worker per DP engine"
            )
        if self.blocks_per_chunk != 1:
            raise ValueError("direct Guard requires blocks_per_chunk=1")
        if config.canonical_layout:
            raise ValueError(
                "direct Guard requires canonical_layout=False "
                "for certified packed pages"
            )
        if self.extra_config.get("cpu_bytes_to_use", 0) != 0:
            raise ValueError("direct Guard requires zero CPU primary bytes")
        if (
            self.extra_config.get("guard_pool_bytes", GUARD_POOL_BYTES)
            != GUARD_POOL_BYTES
        ):
            raise ValueError("direct Guard requires a 124 GiB independent Guard pool")
        for name in ("guard_socket_path", "compatibility_digest"):
            if (
                not isinstance(self.extra_config.get(name), str)
                or not self.extra_config[name]
            ):
                raise ValueError(f"direct Guard requires {name}")
        self._manager = None
        self._worker = None

    def get_manager(self):
        if self._manager is None:
            from .manager import DirectGuardManager

            self._manager = DirectGuardManager(self.config)
        return self._manager

    def get_worker(self, kv_caches):
        if self._worker is None:
            from .worker import DirectGuardWorker

            self._worker = DirectGuardWorker(self.config, kv_caches)
        return self._worker
