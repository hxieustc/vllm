# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import unittest
from types import SimpleNamespace as N

from ._paths import direct_module


class Jobs(unittest.TestCase):
    def test_partial_completions_and_hazards_never_ack_gpu_reuse(self):
        GPUHazards = direct_module("hazards").GPUHazards
        IOJob = direct_module("jobs").IOJob

        j = IOJob(7, "load", "r", {b"a": 3, b"b": 5}, None, None, 16)
        j.add_operation(2, [b"a", b"b"])
        j.complete(2, {b"a": N(success=True)})
        h = GPUHazards("own")
        self.assertFalse(j.releasable(h))
        j.complete(2, {b"b": N(success=False)})
        h.active[99] = {5}
        self.assertFalse(j.releasable(h))
        h.active.clear()
        self.assertTrue(j.releasable(h))
        result = j.result()
        self.assertFalse(result.success)
        self.assertEqual(result.failed_block_ids, (5,))
        self.assertEqual(result.transfer_size, 16)

    def test_duplicate_or_wrong_operation_completion_is_not_accepted(self):
        IOJob = direct_module("jobs").IOJob

        j = IOJob(1, "store", "r", {b"a": 1}, None, None, 8)
        j.add_operation(2, [b"a"])
        with self.assertRaises(ValueError):
            j.complete(2, {b"b": N(success=True)})
        j.complete(2, {b"a": N(success=True)})
        with self.assertRaises(ValueError):
            j.complete(2, {b"a": N(success=True)})


if __name__ == "__main__":
    unittest.main()
