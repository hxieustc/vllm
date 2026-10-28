# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import unittest
from types import SimpleNamespace as N
from typing import Any

import pytest

from ._paths import direct_module

DirectResult: Any


class CompletionTests(unittest.TestCase):
    def test_partial_load_is_acked_with_exact_invalid_gpu_ids(self):
        consume_results = direct_module("completion").consume_results

        calls: list[int] = []
        meta = N(
            mark_completed=calls.append,
            transfer_stats=N(
                load=N(record=lambda *args: None), store=N(record=lambda *args: None)
            ),
        )
        invalid: set[int] = set()
        loads = {7: "request"}
        received = consume_results(
            [DirectResult(7, False, 16, 0.1, "load", (5,), (b"failed",))],
            loads,
            meta,
            invalid,
        )
        self.assertEqual(received, {"request"})
        self.assertEqual(invalid, {5})
        self.assertEqual(calls, [7])
        self.assertFalse(loads)

    def test_hybrid_failure_reports_request_as_well_as_exact_invalid_blocks(self):
        consume_results = direct_module("completion").consume_results

        meta = N(
            mark_completed=lambda i: None,
            transfer_stats=N(load=N(record=lambda *args: None)),
        )
        failed: set[str] = set()
        received = consume_results(
            [DirectResult(7, False, 16, 0.1, "load", (5,), (b"bad",))],
            {7: "hybrid-r"},
            meta,
            set(),
            failed,
        )
        self.assertEqual(received, {"hybrid-r"})
        self.assertEqual(failed, {"hybrid-r"})

    def test_best_effort_failed_put_does_not_become_a_receive_failure(self):
        consume_results = direct_module("completion").consume_results

        calls: list[int] = []
        meta = N(
            mark_completed=calls.append,
            transfer_stats=N(load=None, store=N(record=lambda *a: None)),
        )
        invalid: set[int] = set()
        received = consume_results(
            [DirectResult(9, False, 0, 0.1, "store", (), (b"failed",))],
            {},
            meta,
            invalid,
        )
        self.assertEqual(received, set())
        self.assertEqual(invalid, set())
        self.assertEqual(calls, [9])

    def test_unmapped_failed_load_is_not_silently_acked(self):
        consume_results = direct_module("completion").consume_results

        calls: list[int] = []
        meta = N(mark_completed=calls.append)
        with self.assertRaises(RuntimeError):
            consume_results(
                [DirectResult(1, False, 0, 0.1, "load")], {1: "r"}, meta, set()
            )
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()


@pytest.fixture(autouse=True)
def _direct_symbols(api):
    global DirectResult
    DirectResult = direct_module("jobs").DirectResult
