from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Literal, Tuple

import torch
from minisgl.core import Batch, Req, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import init_logger
from tqdm import tqdm

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend
    from minisgl.models import BaseLLMModel

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def init(
        cls, max_forward_tokens: int, vocab_size: int, device: torch.device
    ) -> GraphCaptureBuffer:
        return GraphCaptureBuffer(
            input_ids=torch.zeros(max_forward_tokens, dtype=torch.int32, device=device),
            out_loc=torch.zeros(max_forward_tokens, dtype=torch.int32, device=device),
            positions=torch.zeros(max_forward_tokens, dtype=torch.int32, device=device),
            logits=torch.empty(max_forward_tokens, vocab_size, dtype=torch.float32, device=device),
        )

    def set_batch(self, batch: Batch) -> None:
        _slice = slice(batch.padded_forward_size)
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]

    def copy_from(self, batch: Batch) -> None:
        _slice = slice(batch.padded_forward_size)
        self.input_ids[_slice] = batch.input_ids
        self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions


GraphPhase = Literal["decode", "verify"]
GraphKey = Tuple[GraphPhase, int]


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))


def mem_GB(size: int) -> str:
    return f"{size / (1024**3):.2f} GiB"


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        logical_max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
        verify_width: int | None,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.stream = stream
        self.device = device
        self.logical_max_seq_len = logical_max_seq_len
        self.verify_width = verify_width
        self._capture_graphs(max_seq_len, vocab_size, model)

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        self.graph_map: Dict[GraphKey, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        self.attn_backend.init_capture_graph(
            max_seq_len=max_seq_len,
            bs_list=self.graph_bs_list,
            verify_width=self.verify_width,
        )

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        max_forward_width = self.verify_width or 1
        self.buffer = GraphCaptureBuffer.init(
            self.max_graph_bs * max_forward_width, vocab_size, self.device
        )

        phases: List[GraphPhase] = ["decode"]
        if self.verify_width is not None:
            # Capture the larger execution first so the shared graph pool is
            # sized for verification before the smaller decode graphs.
            phases.insert(0, "verify")
        capture_keys = [
            (phase, bs) for phase in phases for bs in sorted(self.graph_bs_list, reverse=True)
        ]

        pbar = tqdm(
            capture_keys,
            desc="Preparing for capturing CUDA graphs...",
            unit="graph",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for phase, bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = (
                f"Capturing graphs: phase = {phase:<6} bs = {bs:<3} | "
                f"avail_mem = {mem_GB(free_memory)}"
            )
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = self._make_capture_batch(phase, bs)
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            forward_size = batch.padded_forward_size
            with get_global_ctx().forward_batch(batch):
                self.buffer.logits[:forward_size] = model.forward()
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.buffer.logits[:forward_size] = model.forward()
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[(phase, bs)] = graph

        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def _make_capture_batch(self, phase: GraphPhase, bs: int) -> Batch:
        if phase == "decode":
            return Batch(reqs=[self.dummy_req] * bs, phase="decode")
        assert self.verify_width is not None
        return Batch(
            reqs=[self.dummy_req] * bs,
            phase="verify",
            draft_ids=[torch.empty(0, dtype=torch.int32)] * bs,
            verify_width=self.verify_width,
        )

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        if batch.size > self.max_graph_bs:
            return False
        if batch.is_decode:
            return True
        if not batch.is_verify or self.verify_width is None:
            return False
        return all(
            batch.verification_len(i) <= self.verify_width
            and req.cached_len + self.verify_width <= self.logical_max_seq_len
            for i, req in enumerate(batch.reqs)
        )

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        self.buffer.copy_from(batch)
        phase: GraphPhase = "verify" if batch.is_verify else "decode"
        g = self.graph_map[(phase, batch.padded_size)]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        return self.buffer.logits[: batch.forward_size]

    def pad_batch(self, batch: Batch) -> None:
        use_cuda_graph = self.can_use_cuda_graph(batch)
        if use_cuda_graph and batch.is_verify:
            assert self.verify_width is not None
            batch.verify_width = self.verify_width
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if use_cuda_graph
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        del self.graph_map
        gc.collect()
