# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import unittest
from types import SimpleNamespace as N

from ._paths import direct_module


class Tensor:
    shape = (12, 1228800)
    dtype = "int8"
    device = N(type="cuda", index=1)

    def stride(self, index):
        return (1228800, 1)[index]

    def element_size(self):
        return 1

    def data_ptr(self):
        return 4096


class LayoutTests(unittest.TestCase):
    def config(self):
        return N(
            cache=N(blocks_per_chunk=1),
            parallel=N(world_size=1, tp_size=1),
            worker_kv_bytes_per_block=1228800,
            groups=[N(group_id=0), N(group_id=2)],
        )

    def layout(self):
        PackedLayout = direct_module("layout").PackedLayout

        return PackedLayout(
            self.config(),
            N(
                tensors=[N(tensor=Tensor(), page_size_bytes=1228800)],
                group_data_refs=[
                    [N(tensor_idx=0, page_size_bytes=1228800)],
                    [N(tensor_idx=0, page_size_bytes=1228800)],
                ],
            ),
        )

    def test_certified_vram_region_and_noncontiguous_group_mapping(self):
        layout = self.layout()
        keys = [
            b"a" + (0).to_bytes(4, "big"),
            b"b" + (0).to_bytes(4, "big"),
            b"c" + (2).to_bytes(4, "big"),
        ]
        mapping = layout.map_indices(
            keys, N(block_ids=[1, 7, 4], group_sizes=[2, 1], block_indices=[0, 0])
        )
        self.assertEqual(list(mapping.values()), [1, 7, 4])
        self.assertEqual(layout.region_kwargs()["mem_type"], "VRAM")
        self.assertEqual(layout.region_kwargs()["device_Id"], 1)

    def test_wrong_group_or_reserved_null_or_wrong_cardinality_rejected(self):
        layout = self.layout()
        for keys, ids, sizes in [
            ([b"a" + (2).to_bytes(4, "big")], [1], [1, 0]),
            ([b"a" + (0).to_bytes(4, "big")], [0], [1, 0]),
            ([b"a" + (0).to_bytes(4, "big")], [1, 2], [2, 0]),
        ]:
            with self.assertRaises(ValueError):
                layout.map_indices(
                    keys, N(block_ids=ids, group_sizes=sizes, block_indices=[0, 0])
                )

    def test_unsupported_geometry_fails_instead_of_host_staging(self):
        PackedLayout = direct_module("layout").PackedLayout

        config = self.config()
        config.cache.blocks_per_chunk = 2
        with self.assertRaises(ValueError):
            PackedLayout(config, N(tensors=[]))


if __name__ == "__main__":
    unittest.main()
