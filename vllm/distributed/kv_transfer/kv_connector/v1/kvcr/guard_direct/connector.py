# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retain vLLM's GPU job fences and make partial GET recomputation explicit."""

from typing import TYPE_CHECKING, cast

from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)

from .completion import consume_results
from .router_geometry import (
    validate_router_geometry as attest_router_geometry,
)
from .router_geometry import (
    write_router_geometry_proof,
)

if TYPE_CHECKING:
    from .manager import DirectGuardManager


class DirectConnectorWorker(OffloadingConnectorWorker):
    def __init__(self, spec, vllm_config, kv_cache_config):
        super().__init__(spec, vllm_config, kv_cache_config)
        self._direct_hybrid = len(kv_cache_config.kv_cache_groups) > 1
        self._direct_invalid_block_ids = set()
        self._direct_failed_requests = set()

    def get_finished(self, finished_req_ids):
        assert self.worker is not None
        received = consume_results(
            self.worker.get_finished(),
            self._load_jobs,
            self._connector_worker_meta,
            self._direct_invalid_block_ids,
            self._direct_failed_requests,
        )
        return set(), received

    def take_failed_requests(self):
        ids, self._direct_failed_requests = self._direct_failed_requests, set()
        return ids

    def take_load_errors(self):
        ids, self._direct_invalid_block_ids = self._direct_invalid_block_ids, set()
        # Group-local IDs cannot be mapped unambiguously by the core scheduler.
        # failed_recving is the authoritative hybrid recomputation signal.
        return set() if self._direct_hybrid else ids


class KVCRDirectGuardConnector(OffloadingConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        # Fail before the spec/manager factory can accept advisory hints.
        geometry = attest_router_geometry(
            vllm_config,
            kv_cache_config,
            require_fallback=role == KVConnectorRole.WORKER,
        )
        self.router_geometry_proof = write_router_geometry_proof(
            geometry, vllm_config, worker=role == KVConnectorRole.WORKER
        )
        super().__init__(vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.WORKER:
            # Base construction creates only bookkeeping; GPU worker initialization
            # happens later in register_kv_caches. Reuse its exact derived spec.
            assert self.connector_worker is not None
            self.connector_worker = DirectConnectorWorker(
                self.connector_worker.spec, vllm_config, kv_cache_config
            )

    def update_connector_output(self, connector_output):
        assert self.connector_scheduler is not None
        for request_id in connector_output.failed_recving or ():
            cast(
                "DirectGuardManager", self.connector_scheduler.manager
            ).mark_load_failure(request_id)
        return super().update_connector_output(connector_output)

    def get_transfer_results(self, finished_req_ids):
        assert isinstance(self.connector_worker, DirectConnectorWorker)
        sent, received = self.connector_worker.get_finished(finished_req_ids)
        failed = self.connector_worker.take_failed_requests()
        return KVConnectorTransferResults(
            finished_sending=sent,
            finished_recving=received | failed,
            failed_recving=failed,
        )

    def get_block_ids_with_load_errors(self):
        assert isinstance(self.connector_worker, DirectConnectorWorker)
        return self.connector_worker.take_load_errors()
