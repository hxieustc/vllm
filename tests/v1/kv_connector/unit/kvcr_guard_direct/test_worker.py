# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import unittest
from types import SimpleNamespace as N
from typing import Any

import pytest

from ._paths import direct_module

IOJob: Any
DirectIOActor: Any


class Client:
    def __init__(self):
        self.calls = []
        self.next = 0
        self.results = []

    def deposit(self, refs, **kwargs):
        self.calls.append("store")
        self.next += 1
        return self.next

    def deliver(self, refs, request):
        self.calls.append("load")
        self.next += 1
        return self.next

    def poll_completed(self):
        r, self.results = self.results, []
        return r

    def discard_hint(self, r):
        pass


class WorkerTests(unittest.TestCase):
    def actor(self):
        a = DirectIOActor(N(extra_config={}), N(page_bytes=16))
        a.client = Client()
        a._event = lambda *args, **kwargs: None
        a.visibility = N(complete_reads=lambda: None)
        return a

    def job(self, i, kind, ready=True):
        return IOJob(
            i, kind, "r", {bytes([i]): i + 1}, N(query=lambda: ready), None, 16
        )

    def test_puts_are_not_starved_by_many_ready_get_jobs(self):
        a = self.actor()
        for i in range(17):
            a.jobs[i] = self.job(i, "load")
        a.jobs[40] = self.job(40, "store")
        a._launch()
        self.assertIn("store", a.client.calls)
        self.assertLessEqual(a.client.calls.index("store"), 4)

    def test_producer_dependency_must_complete_before_native_submit(self):
        a = self.actor()
        a.jobs[1] = self.job(1, "store", False)
        a._launch()
        self.assertEqual(a.client.calls, [])

    def test_metadata_backpressure_never_blocks_native_progress(self):
        import msgspec
        import zmq

        a = self.actor()
        a.client.get_stats = lambda: None

        class Socket:
            incoming = [(b"peer", msgspec.msgpack.encode({"id": 1, "kind": "poll"}))]

            def recv_multipart(self, flags=0):
                if self.incoming:
                    return self.incoming.pop()
                raise zmq.Again()

            def send_multipart(self, frames, flags=0):
                if flags != zmq.NOBLOCK:
                    raise AssertionError("Blocking metadata reply")
                raise zmq.Again()

        a._metadata(Socket())
        self.assertEqual(len(a._replies), 1)

    def test_idle_gauges_are_coalesced_instead_of_retained_per_poll(self):
        a = self.actor()
        a.client.get_stats = lambda: N(
            _types={"metric": "gauge"}, _values={"metric": {(): 7}}
        )
        a._stats(force=True)
        a._stats(force=True)
        self.assertEqual(len(a.pending_stats), 1)

    def test_failed_get_result_does_not_release_gpu_while_hazard_remains(self):
        a = self.actor()
        j = self.job(2, "load")
        j.unsent = []
        j.add_operation(77, [b"\x02"])
        a.jobs[2] = j
        a.operation_jobs[77] = j
        a.hazards.active[90] = {3}
        a.client.results = [(77, {b"\x02": N(success=False)})]
        a._poll()
        self.assertTrue(a.completed.empty())
        a.hazards.active.clear()
        a._poll()
        r = a.completed.get_nowait()
        self.assertFalse(r.success)
        self.assertEqual(r.failed_block_ids, (3,))

    def test_actor_selects_its_device_through_accelerator_api(self):
        from unittest.mock import patch

        selected: list[int] = []
        torch = N(accelerator=N(set_device_index=selected.append))
        client = N(get_stats=lambda: None, close=lambda: None)
        actor = DirectIOActor(
            N(extra_config={}),
            N(page_bytes=16, device_index=2),
            client_factory=lambda actor: client,
            visibility=N(),
            endpoint="inproc://guard-direct-device-test",
        )
        actor.stop_requested.set()
        with patch.dict("sys.modules", torch=torch):
            actor.run()
        self.assertEqual(selected, [2])
        self.assertTrue(actor.quiescent)
        self.assertIsNone(actor.error)


if __name__ == "__main__":
    unittest.main()


@pytest.fixture(autouse=True)
def _direct_symbols(api):
    global IOJob, DirectIOActor
    IOJob = direct_module("jobs").IOJob
    DirectIOActor = direct_module("worker").DirectIOActor
