# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU reservations remain held after failed results until each hazard quiesces."""

from typing import Any


class GPUHazards:
    def __init__(self, agent):
        self.agent = agent
        self.active = {}

    def observe(self, error):
        if error.state not in ("uncertain", "quiesced"):
            raise ValueError("Unknown transfer hazard state")
        regions: list[Any] = []
        if getattr(error, "destination_owner", None) == "framework":
            regions.extend(getattr(error, "destination_regions", ()) or ())
        owners = getattr(error, "source_owners", None) or {}
        for key, refs in (getattr(error, "source_blocks", None) or {}).items():
            if owners.get(key) == "framework":
                regions.extend(refs)
        if not regions:
            return
        indices = set()
        for ref in regions:
            if ref.end_point_name != self.agent or ref.label != "":
                raise RuntimeError("Unrecognized framework GPU hazard reference")
            indices.add(int(ref.element_index))
        token = error.op_handle
        if error.state == "uncertain":
            self.active.setdefault(token, set()).update(indices)
        else:
            self.active.pop(token, None)

    def blocks(self, indices):
        return any(set(indices) & held for held in self.active.values())
