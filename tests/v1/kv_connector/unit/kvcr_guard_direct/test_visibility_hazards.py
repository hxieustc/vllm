# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import unittest
from types import SimpleNamespace as N

from ._paths import direct_module


class VisibilityHazards(unittest.TestCase):
    def test_native_ordering_or_supported_explicit_flush_required(self):
        GPUVisibility = direct_module("visibility").GPUVisibility

        calls = []
        v = GPUVisibility(100, 0, lambda: calls.append(1))
        v.complete_reads()
        self.assertFalse(calls)
        v = GPUVisibility(0, 1, lambda: calls.append(1))
        v.complete_reads()
        self.assertEqual(calls, [1])
        with self.assertRaises(RuntimeError):
            GPUVisibility(0, 0, lambda: None)

    def test_uncertain_blocks_hold_all_overlapping_jobs_until_own_quiescence(self):
        GPUHazards = direct_module("hazards").GPUHazards

        h = GPUHazards("own")
        ref = lambda i: N(end_point_name="own", element_index=i, label="")
        error = lambda op, state, indices: N(
            op_handle=op,
            state=state,
            destination_owner="framework",
            destination_regions=[ref(i) for i in indices],
            source_blocks=None,
            source_owners=None,
        )
        h.observe(error(9, "uncertain", [3]))
        h.observe(error(10, "uncertain", [3, 5]))
        self.assertTrue(h.blocks({3}))
        self.assertFalse(h.blocks({4}))
        h.observe(error(9, "quiesced", [3]))
        self.assertTrue(h.blocks({3}))
        h.observe(error(10, "quiesced", [3, 5]))
        self.assertFalse(h.blocks({3, 5}))

    def test_guard_owned_source_uncertainty_is_not_gpu_quiescence(self):
        GPUHazards = direct_module("hazards").GPUHazards

        h = GPUHazards("own")
        h.observe(
            N(
                op_handle=1,
                state="uncertain",
                destination_owner="kvcr",
                destination_regions=[N(element_index=2, end_point_name="own")],
                source_blocks=None,
                source_owners=None,
            )
        )
        self.assertFalse(h.blocks({2}))


if __name__ == "__main__":
    unittest.main()
