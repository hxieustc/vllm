# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import unittest
from types import SimpleNamespace as N
from typing import Any, cast

import pytest

from ._paths import ADAPTER, direct_module

consume_results: Any
DirectResult: Any


def classes():
    class BaseWorker:
        def __init__(self, *args):
            self.worker = None
            self._load_jobs = {}
            self._connector_worker_meta = N(
                mark_completed=lambda i: None,
                transfer_stats=N(
                    load=N(record=lambda *a: None), store=N(record=lambda *a: None)
                ),
            )

    class BaseConnector:
        calls: list[str]

        def update_connector_output(self, output):
            self.calls.append("parent")

    tree = ast.parse((ADAPTER / "connector.py").read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    namespace = dict(
        cast=cast,
        OffloadingConnectorWorker=BaseWorker,
        OffloadingConnector=BaseConnector,
        KVConnectorRole=N(WORKER="worker"),
        consume_results=consume_results,
        KVConnectorTransferResults=lambda **kwargs: N(**kwargs),
    )
    exec(compile(tree, "actual_connector_classes", "exec"), namespace)
    return namespace


class ConnectorContract(unittest.TestCase):
    def test_hybrid_receive_failure_never_emits_forbidden_flat_block_ids(self):
        n = classes()
        w = n["DirectConnectorWorker"](None, None, N(kv_cache_groups=[1, 2]))
        w.worker = N(
            get_finished=lambda: [
                DirectResult(7, False, 0, 0.1, "load", (5,), (b"key",))
            ]
        )
        w._load_jobs = {7: "r"}
        c = n["KVCRDirectGuardConnector"].__new__(n["KVCRDirectGuardConnector"])
        c.connector_worker = w
        transfer = c.get_transfer_results(set())
        self.assertEqual(transfer.failed_recving, {"r"})
        self.assertEqual(transfer.finished_recving, {"r"})
        self.assertEqual(c.get_block_ids_with_load_errors(), set())

    def test_single_group_keeps_exact_failed_block_ids(self):
        n = classes()
        w = n["DirectConnectorWorker"](None, None, N(kv_cache_groups=[1]))
        w._direct_invalid_block_ids = {5}
        self.assertEqual(w.take_load_errors(), {5})

    def test_scheduler_records_failure_before_completed_job_ack(self):
        n = classes()
        c = n["KVCRDirectGuardConnector"].__new__(n["KVCRDirectGuardConnector"])
        c.calls = []
        c.connector_scheduler = N(
            manager=N(mark_load_failure=lambda r: c.calls.append(r))
        )
        c.update_connector_output(N(failed_recving={"r"}))
        self.assertEqual(c.calls, ["r", "parent"])


if __name__ == "__main__":
    unittest.main()


@pytest.fixture(autouse=True)
def _direct_symbols(api):
    global consume_results, DirectResult
    consume_results = direct_module("completion").consume_results
    DirectResult = direct_module("jobs").DirectResult
