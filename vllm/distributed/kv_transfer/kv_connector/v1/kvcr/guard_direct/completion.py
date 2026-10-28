# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bridge safe direct completions to the pinned vLLM invalid-block contract."""


def consume_results(results, load_jobs, metadata, invalid_ids, failed_requests=None):
    finished_recving = set()
    for result in results:
        is_load = result.job_id in load_jobs
        if (result.kind == "load") != is_load:
            raise RuntimeError("Direct result direction/job ownership mismatch")
        if is_load and not result.success:
            if not result.failed_block_ids:
                raise RuntimeError(
                    "Failed GPU load lacks explicit invalid destinations"
                )
            invalid_ids.update(result.failed_block_ids)
            if failed_requests is not None:
                failed_requests.add(load_jobs[result.job_id])
        if result.transfer_size is not None and result.transfer_time is not None:
            stats = (
                metadata.transfer_stats.load
                if is_load
                else metadata.transfer_stats.store
            )
            stats.record(result.transfer_size, result.transfer_time)
        metadata.mark_completed(result.job_id)
        if is_load:
            finished_recving.add(load_jobs.pop(result.job_id))
    return finished_recving
