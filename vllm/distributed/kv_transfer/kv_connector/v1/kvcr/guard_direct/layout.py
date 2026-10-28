# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Certified DSv4 packed pages: physical GPU IDs are never host-cache IDs."""


class PackedLayout:
    def __init__(self, config, caches):
        if config.parallel.world_size != 1 or config.parallel.tp_size != 1:
            raise ValueError("Direct Guard currently certifies TP1 only")
        if config.cache.blocks_per_chunk != 1 or len(caches.tensors) != 1:
            raise ValueError(
                "Direct Guard requires one packed tensor and one GPU block per chunk"
            )
        item = caches.tensors[0]
        tensor = item.tensor
        page = int(item.page_size_bytes)
        if (
            tensor.device.type != "cuda"
            or tensor.element_size() != 1
            or len(tensor.shape) != 2
            or int(tensor.shape[1]) != page
            or tensor.stride(0) < page
            or tensor.stride(1) != 1
            or page != config.worker_kv_bytes_per_block
        ):
            raise ValueError(
                "GPU packed page geometry differs from configured offload geometry"
            )
        if len(caches.group_data_refs) != len(config.groups):
            raise ValueError("GPU/group geometry mismatch")
        for refs in caches.group_data_refs:
            if (
                len(refs) != 1
                or refs[0].tensor_idx != 0
                or refs[0].page_size_bytes != page
            ):
                raise ValueError(
                    "Every direct group must reference its complete packed page"
                )
        self.tensor = tensor
        self.page_bytes = page
        self.count = int(tensor.shape[0])
        self.group_ids = tuple(g.group_id for g in config.groups)
        self.device_index = int(tensor.device.index)

    def region_kwargs(self):
        return dict(
            mem_type="VRAM",
            device_Id=self.device_index,
            addr=self.tensor.data_ptr(),
            stride=self.tensor.stride(0),
            size=self.page_bytes,
            count=self.count,
            label="",
        )

    def map_indices(self, keys, gpu_spec):
        keys = tuple(bytes(k) for k in keys)
        ids = tuple(int(i) for i in gpu_spec.block_ids)
        sizes = tuple(int(n) for n in gpu_spec.group_sizes)
        if (
            len(keys) != len(ids)
            or len(sizes) != len(self.group_ids)
            or sum(sizes) != len(ids)
        ):
            raise ValueError("Direct key/GPU page cardinality mismatch")
        if any(n < 0 for n in sizes) or len(set(keys)) != len(keys):
            raise ValueError("Negative group size or duplicate direct key")
        offset = 0
        for group, size in zip(self.group_ids, sizes):
            for key in keys[offset : offset + size]:
                if len(key) < 5 or int.from_bytes(key[-4:], "big") != group:
                    raise ValueError(
                        "Direct key order differs from original KV group order"
                    )
            offset += size
        # vLLM reserves its physical page zero as the null block.
        if any(i <= 0 or i >= self.count for i in ids):
            raise ValueError("GPU block is null or outside the registered tensor")
        return dict(zip(keys, ids))
