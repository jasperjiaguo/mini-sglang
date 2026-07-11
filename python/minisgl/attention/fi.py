from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from itertools import accumulate
from typing import TYPE_CHECKING, Dict, List, Literal, Tuple

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.env import ENV
from minisgl.utils import div_even, init_logger

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    )
    from minisgl.models import ModelConfig


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << math.ceil(math.log2(n))


logger = init_logger(__name__)


@dataclass
class FICaptureData(BaseCaptureData):
    @property
    def one_tensor(self) -> torch.Tensor:
        return self.seq_lens

    @property
    def indices(self) -> torch.Tensor:
        return self.page_table


@dataclass
class _FIMetadataStagingBuffer:
    seq_lens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    completion_event: torch.cuda.Event
    pending: bool = False

    @classmethod
    def create(cls, capacity: int) -> _FIMetadataStagingBuffer:
        return cls(
            seq_lens=torch.empty(capacity, dtype=torch.int32, pin_memory=True),
            cu_seqlens_q=torch.empty(capacity + 1, dtype=torch.int32, pin_memory=True),
            cu_seqlens_k=torch.empty(capacity + 1, dtype=torch.int32, pin_memory=True),
            completion_event=torch.cuda.Event(),
        )


@dataclass
class FIMetadata(BaseAttnMetadata):
    # fmt: off
    cu_seqlens_q_cpu:   torch.Tensor  # on cpu
    cu_seqlens_k_cpu:   torch.Tensor  # on cpu
    cu_seqlens_q_gpu:   torch.Tensor  # on gpu
    indices:            torch.Tensor  # on gpu
    last_page_len_cpu:  torch.Tensor  # on cpu
    num_qo_heads:       int
    num_kv_heads:       int
    head_dim:           int
    page_size:          Literal[1] # currently only support page_size=1
    pos_encoding_mode:  str
    seq_lens_cpu:       torch.Tensor  # on cpu
    dtype:              torch.dtype
    wrapper:            BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper
    staging_buffer:     _FIMetadataStagingBuffer = field(repr=False)
    initialized:        bool = False
    # fmt: on

    def __post_init__(self) -> None:
        assert self.page_size == 1, "Currently only page_size=1 is supported."
        assert (
            self.cu_seqlens_k_cpu.is_cpu
            and self.cu_seqlens_q_cpu.is_cpu
            and self.cu_seqlens_q_gpu.is_cuda
            and self.indices.is_cuda
            and self.last_page_len_cpu.is_cpu
            and self.seq_lens_cpu.is_cpu
        )

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1


class FlashInferBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device
        self.float_workspace_buffer = torch.empty(
            ENV.FLASHINFER_WORKSPACE_SIZE.value,
            dtype=torch.uint8,
            device=self.device,
        )
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 is slow, use fa2 instead
        )
        self.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            use_tensor_cores=self.use_tensor_cores,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 is slow, use fa2 instead
        )

        # NOTE: some hack to reuse the int_workspace_buffer
        self.int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer
        self.decode_wrappers._int_workspace_buffer = self.int_workspace_buffer

        # initialize some data members
        tp_size = get_tp_info().size
        self.qo_head_local = div_even(self.config.num_qo_heads, tp_size)
        self.kv_head_local = div_even(self.config.num_kv_heads, tp_size, allow_replicate=True)

        self.cached_ones_cpu: torch.Tensor = torch.tensor([], dtype=torch.int32, pin_memory=True)
        self._metadata_staging_buffer_index = 0
        self._metadata_staging_buffers = [
            _FIMetadataStagingBuffer.create(1),
            _FIMetadataStagingBuffer.create(1),
        ]
        # for cuda graph
        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self.graph_wrappers: Dict[
            Tuple[str, int],
            CUDAGraphBatchDecodeWithPagedKVCacheWrapper | BatchPrefillWithPagedKVCacheWrapper,
        ] = {}
        self.capture: FICaptureData | None = None
        self.last_event = torch.cuda.Event()
        self.last_event.record()

    def _initialize_metadata_once(self, metadata: FIMetadata) -> None:
        if metadata.initialized:
            return

        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        metadata.initialized = True
        # FlashInfer planning reuses a pinned host staging buffer and launches an
        # async H2D copy. Wait here before the next plan mutates that host buffer.
        self.last_event.synchronize()
        if isinstance(metadata.wrapper, BatchDecodeWithPagedKVCacheWrapper):
            metadata.wrapper.plan(
                indptr=metadata.cu_seqlens_k_cpu,
                indices=metadata.indices,
                last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                data_type=metadata.dtype,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
            )
        else:
            metadata.wrapper.plan(
                qo_indptr=metadata.cu_seqlens_q_cpu,
                paged_kv_indptr=metadata.cu_seqlens_k_cpu,
                paged_kv_indices=metadata.indices,
                paged_kv_last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim_qk=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
                causal=True,
            )
        self.last_event.record()
        metadata.staging_buffer.completion_event.record()
        metadata.staging_buffer.pending = True

    def _get_ones_cpu(self, bs: int) -> torch.Tensor:
        if bs <= len(self.cached_ones_cpu):
            return self.cached_ones_cpu[:bs]
        # padding to next pow of 2
        next_len = _next_power_of_2(bs)
        self.cached_ones_cpu = torch.ones(next_len, dtype=torch.int32, pin_memory=True)
        return self.cached_ones_cpu[:bs]

    def _get_metadata_staging_buffer(self, bs: int) -> _FIMetadataStagingBuffer:
        index = self._metadata_staging_buffer_index
        self._metadata_staging_buffer_index = (index + 1) % len(self._metadata_staging_buffers)
        buffer = self._metadata_staging_buffers[index]
        if buffer.pending:
            buffer.completion_event.synchronize()
            buffer.pending = False
        if bs > len(buffer.seq_lens):
            buffer = _FIMetadataStagingBuffer.create(_next_power_of_2(bs))
            self._metadata_staging_buffers[index] = buffer
        return buffer

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        def _flatten_cache(cache: torch.Tensor) -> torch.Tensor:  # treat page = 1
            return cache.view(-1, 1, cache.shape[2], cache.shape[3])

        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        self._initialize_metadata_once(metadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))
        kv_cache = (_flatten_cache(kv_cache[0]), _flatten_cache(kv_cache[1]))
        return metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs

        padded_size = len(reqs)
        staging = self._get_metadata_staging_buffer(padded_size)
        seq_len_cpu = staging.seq_lens[:padded_size]
        cu_seqlens_q_cpu = staging.cu_seqlens_q[: padded_size + 1]
        cu_seqlens_k_cpu = staging.cu_seqlens_k[: padded_size + 1]
        cu_seqlens_q_cpu[0] = 0
        cu_seqlens_k_cpu[0] = 0
        seqlens_q = [batch.forward_extend_len(i) for i in range(padded_size)]
        seqlens_k = [batch.forward_device_len(i) for i in range(padded_size)]
        seq_len_cpu.numpy()[:] = seqlens_k
        cu_seqlens_k_cpu.numpy()[:] = [0, *accumulate(seqlens_k)]
        max_seqlen_q = max(seqlens_q)
        no_cache_hit = all(req.cached_len == 0 for req in reqs)

        device = self.device
        if max_seqlen_q == 1:  # decode with all extend_len = 1
            cu_seqlens_q_cpu.numpy()[:] = range(padded_size + 1)
        elif no_cache_hit:  # prefill with no cache hit
            cu_seqlens_q_cpu.copy_(cu_seqlens_k_cpu)
        else:  # normal extend prefill, with partial cache hit
            cu_seqlens_q_cpu.numpy()[:] = [0, *accumulate(seqlens_q)]

        page_table = get_global_ctx().page_table
        batch.attn_metadata = FIMetadata(
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            cu_seqlens_k_cpu=cu_seqlens_k_cpu,
            cu_seqlens_q_gpu=cu_seqlens_q_cpu.to(device, non_blocking=True),
            indices=torch.cat(
                [
                    page_table[req.table_idx, : batch.forward_device_len(i)]
                    for i, req in enumerate(reqs)
                ]
            ),
            last_page_len_cpu=self._get_ones_cpu(padded_size),
            num_qo_heads=self.qo_head_local,
            num_kv_heads=self.kv_head_local,
            head_dim=self.config.head_dim,
            page_size=1,
            pos_encoding_mode="NONE",
            seq_lens_cpu=seq_len_cpu,
            dtype=self.kvcache.dtype,
            wrapper=self.decode_wrappers if batch.is_decode else self.prefill_wrapper,
            staging_buffer=staging,
        )

    def init_capture_graph(
        self, max_seq_len: int, bs_list: List[int], verify_width: int | None = None
    ) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = FICaptureData.create(max_bs, max_seq_len, self.kvcache.device)
        capture.page_table = capture.page_table.view(-1)  # use 1D as ragged indices
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    @cached_property
    def use_tensor_cores(self) -> bool:
        if (overriden_value := ENV.FLASHINFER_USE_TENSOR_CORES.value) is not None:
            logger.warning(f"Overriding FlashInfer tensor core usage to {overriden_value}")
            return overriden_value
        GQA = self.config.num_qo_heads // self.config.num_kv_heads
        return GQA >= 4

    def prepare_for_capture(self, batch: Batch) -> None:
        from flashinfer import (
            BatchPrefillWithPagedKVCacheWrapper,
            CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
        )

        bs = batch.size
        key = (batch.phase, bs)
        assert batch.is_decode or batch.is_verify, (
            "Only decode and verification graphs are supported."
        )
        assert bs in self.capture_bs and key not in self.graph_wrappers and self.capture
        capture = self.capture
        if batch.is_decode:
            wrapper = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
                self.float_workspace_buffer,
                kv_layout="NHD",
                use_tensor_cores=self.use_tensor_cores,
                indptr_buffer=capture.cu_seqlens_k[: bs + 1],
                indices_buffer=capture.indices,
                last_page_len_buffer=capture.one_tensor[:bs],
            )
            wrapper._backend = "fa2"
        else:
            wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self.float_workspace_buffer,
                kv_layout="NHD",
                use_cuda_graph=True,
                qo_indptr_buf=capture.cu_seqlens_q[: bs + 1],
                paged_kv_indptr_buf=capture.cu_seqlens_k[: bs + 1],
                paged_kv_indices_buf=capture.indices,
                paged_kv_last_page_len_buf=capture.one_tensor[:bs],
                backend="fa2",
            )
        wrapper._int_workspace_buffer = self.int_workspace_buffer
        self.graph_wrappers[key] = wrapper
        self.prepare_metadata(batch)
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        metadata.wrapper = wrapper
        self._initialize_metadata_once(metadata)

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FIMetadata) and not metadata.initialized
        assert self.capture is not None and bs in self.capture_bs
        metadata.wrapper = self.graph_wrappers[(batch.phase, bs)]
        self._initialize_metadata_once(metadata)
