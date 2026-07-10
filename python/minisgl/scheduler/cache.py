from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, Req
from minisgl.kvcache import BaseCacheHandle, MatchResult, create_prefix_cache
from minisgl.utils import div_ceil

if TYPE_CHECKING:
    from .utils import PendingReq


@dataclass
class _PageTableStagingBuffer:
    table_indices_host: torch.Tensor
    positions_host: torch.Tensor
    table_indices_device: torch.Tensor
    positions_device: torch.Tensor
    completion_event: torch.cuda.Event | None
    pending: bool = False

    @classmethod
    def create(cls, capacity: int, device: torch.device) -> _PageTableStagingBuffer:
        pin_memory = device.type == "cuda"
        return cls(
            table_indices_host=torch.empty(capacity, dtype=torch.int64, pin_memory=pin_memory),
            positions_host=torch.empty(capacity, dtype=torch.int64, pin_memory=pin_memory),
            table_indices_device=torch.empty(capacity, dtype=torch.int64, device=device),
            positions_device=torch.empty(capacity, dtype=torch.int64, device=device),
            completion_event=torch.cuda.Event() if device.type == "cuda" else None,
        )


class CacheManager:
    def __init__(self, num_pages: int, page_size: int, page_table: torch.Tensor, type: str):
        # The `_free_slots` follows a page-aligned manner. For example, if page_size = 2,
        # the `_free_slots` may look like [0, 2, 4, 6, ...], and each slot represents a page.
        device = page_table.device
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        self.prefix_cache = create_prefix_cache(device=device, type=type)
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size
        self._page_table_staging_buffer_index = 0
        self._page_table_staging_buffers = [
            _PageTableStagingBuffer.create(1, device),
            _PageTableStagingBuffer.create(1, device),
        ]

    def match_req(self, req: PendingReq) -> MatchResult:
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size

    def lock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=True)

    def allocate_paged(self, batch: Batch) -> None:
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for i, req in enumerate(batch.reqs):
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(batch.allocated_device_len(i), self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            allocated = self._page_to_token(self._allocate(needed_pages))
            self._write_page_table(allocated, allocation_info)

    def allocate_fused_verify(self, batch: Batch) -> tuple[torch.Tensor, List[int]]:
        """Reserve real verification rows without staging page-table indices."""
        assert self.page_size == 1 and batch.is_verify
        allocation_offsets: List[int] = []
        needed_pages = 0
        for i, req in enumerate(batch.reqs):
            allocation_offsets.append(needed_pages)
            verification_len = batch.verification_len(i)
            assert batch.allocated_device_len(i) == req.cached_len + verification_len
            needed_pages += verification_len
        assert needed_pages > 0
        return self._allocate(needed_pages), allocation_offsets

    def _write_page_table(
        self,
        allocated: torch.Tensor,
        allocation_info: List[Tuple[int, int, int]],
    ) -> None:
        needed_tokens = len(allocated)
        index = self._page_table_staging_buffer_index
        self._page_table_staging_buffer_index = (index + 1) % len(self._page_table_staging_buffers)
        buffer = self._page_table_staging_buffers[index]
        if buffer.pending:
            assert buffer.completion_event is not None
            buffer.completion_event.synchronize()
            buffer.pending = False
        if needed_tokens > len(buffer.table_indices_host):
            capacity = 1 << (needed_tokens - 1).bit_length()
            buffer = _PageTableStagingBuffer.create(capacity, self.device)
            self._page_table_staging_buffers[index] = buffer

        offset = 0
        for table_idx, first_page, last_page in allocation_info:
            first_pos = first_page * self.page_size
            last_pos = last_page * self.page_size
            length = last_pos - first_pos
            buffer.table_indices_host[offset : offset + length].fill_(table_idx)
            torch.arange(
                first_pos,
                last_pos,
                out=buffer.positions_host[offset : offset + length],
            )
            offset += length
        assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."

        buffer.table_indices_device[:offset].copy_(
            buffer.table_indices_host[:offset], non_blocking=True
        )
        buffer.positions_device[:offset].copy_(buffer.positions_host[:offset], non_blocking=True)
        self.page_table[buffer.table_indices_device[:offset], buffer.positions_device[:offset]] = (
            allocated
        )
        if buffer.completion_event is not None:
            buffer.completion_event.record(torch.cuda.current_stream(self.device))
            buffer.pending = True

    def free_req_suffix(self, req: Req, start: int, end: int) -> None:
        assert self.page_size == 1
        assert req.cached_len <= start <= end
        self._free(self.page_table[req.table_idx, start:end])

    def free_req_suffixes(
        self,
        reqs: List[Req],
        starts: List[int],
        ends: List[int],
    ) -> None:
        assert self.page_size == 1
        assert len(reqs) == len(starts) == len(ends)
        suffixes = []
        for req, start, end in zip(reqs, starts, ends, strict=True):
            assert req.cached_len <= start <= end
            if start < end:
                suffixes.append(self.page_table[req.table_idx, start:end])
        if suffixes:
            self._free_many(suffixes)

    def cache_req(self, req: Req, *, finished: bool) -> None:
        # ==================================== valid cache region ====================================
        # [0, req.cached_len)                       This part is valid for attention kernel read/write.
        # [0, old_handle.cached_len)                This part is in the prefix cache before prefill.
        # [old_handle.cached_len, req.cached_len)   This part is allocated by cache manager for this request.
        # ================================== allocated cache region ==================================
        # [old_handle.cached_len, cached_len)       This part was not in the prefix cache when prefill,
        #                                           but later cached by other requests.
        #                                           We must free them to avoid memory leak.
        # [cached_len, new_handle.cached_len)       This part is newly inserted into the prefix cache.
        # [new_handle.cached_len, req.cached_len)   This part is tailing part that can not inserted into the prefix cache.
        #                                           We should free it if the request has finished.
        insert_ids = req.input_ids[: req.cached_len]
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        old_handle = req.cache_handle
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        # unlock until all operations on handle is done
        self.unlock(old_handle)
        # this part is already in the prefix cache, free it
        self._free(page_indices[old_handle.cached_len : cached_len])
        if finished:  # this tail part should be freed
            self._free(page_indices[new_handle.cached_len :])
        else:  # keep the tail part, update the handle
            req.cache_handle = new_handle
            self.lock(new_handle)

    def check_integrity(self) -> None:
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self.free_slots)}) +"
                f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1:
            assert torch.all(self.free_slots % self.page_size == 0)

    @contextmanager
    def lazy_free_region(self):
        def lazy_free(indices: torch.Tensor) -> None:
            if len(indices):
                lazy_free_list.append(indices[:: self.page_size])

        def lazy_free_many(indices: List[torch.Tensor]) -> None:
            lazy_free_list.extend(index[:: self.page_size] for index in indices if len(index))

        lazy_free_list: List[torch.Tensor] = []
        try:
            self._free = lazy_free
            self._free_many = lazy_free_many
            yield
        finally:
            del self._free
            del self._free_many
            if lazy_free_list:
                self.free_slots = torch.cat([self.free_slots] + lazy_free_list)

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        if needed_pages > (free_pages := len(self.free_slots)):
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
            self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
            assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])

    def _free_many(self, indices: List[torch.Tensor]) -> None:
        nonempty = [index[:: self.page_size] for index in indices if len(index)]
        if nonempty:
            self.free_slots = torch.cat([self.free_slots] + nonempty)

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()
