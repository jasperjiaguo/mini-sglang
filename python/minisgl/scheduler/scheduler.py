from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .speculative import _create_speculator
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


@dataclass
class _DraftStagingBuffer:
    token_ids_host: torch.Tensor
    locations_host: torch.Tensor
    token_ids_device: torch.Tensor
    locations_device: torch.Tensor
    completion_event: torch.cuda.Event | None
    pending: bool = False

    @classmethod
    def create(
        cls,
        capacity: int,
        device: torch.device,
        *,
        pin_memory: bool,
    ) -> _DraftStagingBuffer:
        return cls(
            token_ids_host=torch.empty(
                capacity,
                dtype=torch.int32,
                pin_memory=pin_memory,
            ),
            locations_host=torch.empty(
                capacity,
                dtype=torch.int64,
                pin_memory=pin_memory,
            ),
            token_ids_device=torch.empty(capacity, dtype=torch.int32, device=device),
            locations_device=torch.empty(capacity, dtype=torch.int64, device=device),
            completion_event=torch.cuda.Event() if device.type == "cuda" else None,
        )


@dataclass
class _MappingStagingBuffer:
    positions_host: torch.Tensor
    req_mapping_host: torch.Tensor
    write_mapping_host: torch.Tensor
    write_positions_host: torch.Tensor
    positions_device: torch.Tensor
    positions_index_device: torch.Tensor
    req_mapping_device: torch.Tensor
    write_mapping_device: torch.Tensor
    write_positions_device: torch.Tensor
    completion_event: torch.cuda.Event | None
    pending: bool = False

    @classmethod
    def create(
        cls,
        forward_capacity: int,
        request_capacity: int,
        device: torch.device,
        *,
        pin_memory: bool,
    ) -> _MappingStagingBuffer:
        return cls(
            positions_host=torch.empty(forward_capacity, dtype=torch.int32, pin_memory=pin_memory),
            req_mapping_host=torch.empty(
                forward_capacity, dtype=torch.int64, pin_memory=pin_memory
            ),
            write_mapping_host=torch.empty(
                request_capacity, dtype=torch.int64, pin_memory=pin_memory
            ),
            write_positions_host=torch.empty(
                request_capacity, dtype=torch.int64, pin_memory=pin_memory
            ),
            positions_device=torch.empty(forward_capacity, dtype=torch.int32, device=device),
            positions_index_device=torch.empty(forward_capacity, dtype=torch.int64, device=device),
            req_mapping_device=torch.empty(forward_capacity, dtype=torch.int64, device=device),
            write_mapping_device=torch.empty(request_capacity, dtype=torch.int64, device=device),
            write_positions_device=torch.empty(request_capacity, dtype=torch.int64, device=device),
            completion_event=torch.cuda.Event() if device.type == "cuda" else None,
        )


@dataclass
class _PendingTokenStagingBuffer:
    destinations_host: torch.Tensor
    source_indices_host: torch.Tensor
    destinations_device: torch.Tensor
    source_indices_device: torch.Tensor
    completion_event: torch.cuda.Event | None
    pending: bool = False

    @classmethod
    def create(
        cls,
        capacity: int,
        device: torch.device,
        *,
        pin_memory: bool,
    ) -> _PendingTokenStagingBuffer:
        return cls(
            destinations_host=torch.empty(capacity, dtype=torch.int64, pin_memory=pin_memory),
            source_indices_host=torch.empty(capacity, dtype=torch.int64, pin_memory=pin_memory),
            destinations_device=torch.empty(capacity, dtype=torch.int64, device=device),
            source_indices_device=torch.empty(capacity, dtype=torch.int64, device=device),
            completion_event=torch.cuda.Event() if device.type == "cuda" else None,
        )


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.speculator = _create_speculator(config)
        self.speculative_overlap_enabled = (
            self.speculator is not None and not ENV.DISABLE_OVERLAP_SCHEDULING
        )
        configured_graph_batch_size = (
            max(config.cuda_graph_bs) if config.cuda_graph_bs else config.cuda_graph_max_bs
        )
        self.speculative_overlap_batch_size = (
            configured_graph_batch_size
            if configured_graph_batch_size is not None and configured_graph_batch_size > 0
            else None
        )
        verify_width = (
            self.speculator.cuda_graph_verify_width if self.speculator is not None else None
        )
        self.engine = Engine(config, cuda_graph_verify_width=verify_width)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.discarded_inflight_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        self._draft_staging_buffer_index = 0
        self._draft_staging_buffers: List[_DraftStagingBuffer] = []
        self._pending_token_staging_buffer_index = 0
        self._pending_token_staging_buffers: List[_PendingTokenStagingBuffer] = []
        if verify_width is not None:
            draft_capacity = config.max_running_req * (verify_width - 1)
            self._draft_staging_buffers = [
                _DraftStagingBuffer.create(
                    draft_capacity,
                    self.device,
                    pin_memory=True,
                )
                for _ in range(2)
            ]
            self._pending_token_staging_buffers = [
                _PendingTokenStagingBuffer.create(
                    config.max_running_req,
                    self.device,
                    pin_memory=True,
                )
                for _ in range(2)
            ]
        mapping_capacity = max(
            config.max_extend_tokens,
            config.max_running_req * (verify_width or 1),
        )
        self._mapping_staging_buffer_index = 0
        self._mapping_staging_buffers = [
            _MappingStagingBuffer.create(
                mapping_capacity,
                config.max_running_req,
                self.device,
                pin_memory=True,
            )
            for _ in range(2)
        ]
        self.profiler = self._create_torch_profiler()
        # self.config = config

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        if self.speculator is not None:
            self.speculator.log_stats(logger)
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Speculative verification can accept a variable number of tokens, so
        # the same request cannot be scheduled again until its prior result is
        # reconciled. Keep overlap by running a disjoint request group while
        # processing the previous group on the scheduler stream.
        if (
            self.speculative_overlap_enabled
            and last_data is not None
            and any(isinstance(req, ChunkedReq) for req in last_data[0].batch.reqs)
        ):
            # A chunk continuation reuses the same request table. Reconcile it
            # before PrefillManager can schedule the continuation.
            self._process_last_data(last_data)
            last_data = None
        inflight_reqs = (
            set(last_data[0].batch.reqs)
            if self.speculative_overlap_enabled and last_data is not None
            else set()
        )
        if self.profiler is None:
            forward_input = self._schedule_next_batch(inflight_reqs)
        else:
            with torch.profiler.record_function("minisgl_schedule_prepare"):
                forward_input = self._schedule_next_batch(inflight_reqs)
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        ongoing_reqs = set(ongoing_data[0].batch.reqs) if ongoing_data is not None else set()
        if self.profiler is None:
            self._process_last_data(last_data, ongoing_reqs)
        else:
            with torch.profiler.record_function("minisgl_process_result"):
                self._process_last_data(last_data, ongoing_reqs)
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        if self.profiler is None:
            forward_input = self._schedule_next_batch()
        else:
            with torch.profiler.record_function("minisgl_schedule_prepare"):
                forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        if self.profiler is None:
            self._process_last_data(ongoing_data)
        else:
            with torch.profiler.record_function("minisgl_process_result"):
                self._process_last_data(ongoing_data)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        if self.speculator is not None:
            self.speculator.log_stats(logger)
        torch.cuda.synchronize(self.device)
        if self.profiler is not None:
            self.profiler.stop()
        self.sync_all_ranks()
        self.engine.shutdown()

    def _create_torch_profiler(self):
        output_dir = os.environ.get("MINISGL_TORCH_PROFILE_DIR")
        if not output_dir:
            return None
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        wait_steps = int(os.environ.get("MINISGL_TORCH_PROFILE_WAIT_STEPS", "0"))
        warmup_steps = int(os.environ.get("MINISGL_TORCH_PROFILE_WARMUP_STEPS", "5"))
        active_steps = int(os.environ.get("MINISGL_TORCH_PROFILE_ACTIVE_STEPS", "100"))
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=wait_steps,
                warmup=warmup_steps,
                active=active_steps,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        )
        profiler.start()
        logger.info_rank0(
            "Torch profiler enabled: dir=%s wait=%d warmup=%d active=%d",
            output_dir,
            wait_steps,
            warmup_steps,
            active_steps,
        )
        return profiler

    def _profile_region(self, name: str):
        if getattr(self, "profiler", None) is None:
            return nullcontext()
        return torch.profiler.record_function(name)

    def _process_last_data(
        self,
        last_data: ForwardData | None,
        inflight_reqs: Set[Req] | None = None,
    ) -> None:
        if last_data is None:
            return

        batch, (next_tokens_gpu, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        with self._profile_region("minisgl_result_wait_copy"):
            copy_done.synchronize()
        if batch.is_verify:
            self._process_verify_data(batch, next_tokens_cpu, next_tokens_gpu)
            return

        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue
                if req in self.discarded_inflight_reqs:
                    self.discarded_inflight_reqs.remove(req)
                    self._free_req_resources(req)
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                has_inflight = req in (inflight_reqs or set())
                eos_hit = not req.sampling_params.ignore_eos and next_token == self.eos_token_id
                # Engine.forward_batch reserves the next output position before
                # the prior overlapped result is reconciled. That reservation
                # must not make the prior token look like the max-length token.
                finished = (not req.can_decode and not has_inflight) or eos_hit
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    if eos_hit and has_inflight:
                        # The extra forward was launched before its predecessor
                        # revealed EOS. Wait for that result, discard it, then
                        # release the request resources.
                        self.discarded_inflight_reqs.add(req)
                    else:
                        self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        self.send_result(reply)

    def _process_verify_data(
        self,
        batch: Batch,
        target_tokens: torch.Tensor,
        target_tokens_gpu: torch.Tensor | None = None,
    ) -> None:
        assert self.speculator is not None and batch.is_verify
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        pending_destinations: List[int] = []
        pending_sources: List[int] = []
        with self._profile_region("minisgl_verify_request_reconcile"):
            with self.cache_manager.lazy_free_region():
                with self._profile_region("minisgl_verify_acceptance"):
                    acceptances = self.speculator.verify_batch(batch, target_tokens)
                    target_starts: List[int] = []
                    token_rows: List[torch.Tensor] = []
                    emitted_rows: List[List[int]] = []
                    accepted_lengths: List[int] = []
                    new_cached_lengths: List[int] = []
                    eos_hits: List[bool] = []
                    offset = 0
                    for req, acceptance in zip(batch.reqs, acceptances, strict=True):
                        target_starts.append(offset)
                        offset += batch.forward_extend_len(len(target_starts) - 1)
                        token_ids = acceptance.token_ids
                        emitted = acceptance.emitted_token_ids

                        eos_hit = False
                        if not req.sampling_params.ignore_eos and self.eos_token_id in emitted:
                            emitted = emitted[: emitted.index(self.eos_token_id) + 1]
                            token_ids = token_ids[: len(emitted)]
                            eos_hit = True

                        accepted_len = len(token_ids)
                        assert 0 < accepted_len <= req.remain_len
                        token_rows.append(token_ids)
                        emitted_rows.append(emitted)
                        accepted_lengths.append(accepted_len)
                        new_cached_lengths.append(req.cached_len + accepted_len)
                        eos_hits.append(eos_hit)
                    assert offset == len(target_tokens)

                with self._profile_region("minisgl_verify_cache_reconcile"):
                    # Verification stored KV for the pending token and every real
                    # draft. Keep the accepted prefix before the new pending token
                    # and return all rejected suffix pages in one operation.
                    self.cache_manager.free_req_suffixes(
                        batch.reqs,
                        new_cached_lengths,
                        [batch.allocated_device_len(i) for i in range(batch.size)],
                    )

                with self._profile_region("minisgl_verify_host_update"):
                    row_width = self.token_pool.size(1)
                    for req, token_ids, emitted, accepted_len, new_cached_len, target_start in zip(
                        batch.reqs,
                        token_rows,
                        emitted_rows,
                        accepted_lengths,
                        new_cached_lengths,
                        target_starts,
                        strict=True,
                    ):
                        output_start = req.device_len
                        req.cached_len = new_cached_len
                        req.device_len += accepted_len
                        req.append_host(token_ids)
                        self.speculator.update_history(req, emitted)
                        assert req.cached_len + 1 == req.device_len == len(req.input_ids)

                        if target_tokens_gpu is None:
                            output = self.token_pool[req.table_idx, output_start : req.device_len]
                            output.copy_(token_ids.pin_memory(), non_blocking=True)
                        else:
                            pending_destinations.append(
                                req.table_idx * row_width + req.device_len - 1
                            )
                            pending_sources.append(target_start + accepted_len - 1)

                with self._profile_region("minisgl_verify_reply_metrics"):
                    for i, (req, emitted, accepted_len, eos_hit, acceptance) in enumerate(
                        zip(
                            batch.reqs,
                            emitted_rows,
                            accepted_lengths,
                            eos_hits,
                            acceptances,
                            strict=True,
                        )
                    ):
                        finished = not req.can_decode or eos_hit
                        for j, token_id in enumerate(emitted):
                            reply.append(
                                DetokenizeMsg(
                                    uid=req.uid,
                                    next_token=token_id,
                                    finished=finished and j == accepted_len - 1,
                                )
                            )

                        accepted_drafts = min(acceptance.accepted_drafts, accepted_len)
                        self.speculator.record_verification(batch, i, accepted_drafts)
                        if finished and req not in self.finished_reqs:
                            self.decode_manager.remove_req(req)
                            self._free_req_resources(req)
                            new_finished_reqs.add(req)

        with self._profile_region("minisgl_verify_pending_scatter"):
            if target_tokens_gpu is not None:
                self._scatter_pending_tokens(
                    pending_destinations,
                    pending_sources,
                    target_tokens_gpu,
                )
        self.finished_reqs = new_finished_reqs
        with self._profile_region("minisgl_verify_send_result"):
            self.send_result(reply)

    def _scatter_pending_tokens(
        self,
        destinations: List[int],
        source_indices: List[int],
        target_tokens_gpu: torch.Tensor,
    ) -> None:
        assert len(destinations) == len(source_indices)
        if not destinations:
            return
        assert self._pending_token_staging_buffers
        buffer = self._pending_token_staging_buffers[self._pending_token_staging_buffer_index]
        self._pending_token_staging_buffer_index = (
            self._pending_token_staging_buffer_index + 1
        ) % len(self._pending_token_staging_buffers)
        if buffer.pending:
            assert buffer.completion_event is not None
            buffer.completion_event.synchronize()

        size = len(destinations)
        assert size <= len(buffer.destinations_host)
        buffer.destinations_host[:size].numpy()[:] = destinations
        buffer.source_indices_host[:size].numpy()[:] = source_indices
        buffer.destinations_device[:size].copy_(buffer.destinations_host[:size], non_blocking=True)
        buffer.source_indices_device[:size].copy_(
            buffer.source_indices_host[:size], non_blocking=True
        )
        self.token_pool.view(-1).index_copy_(
            0,
            buffer.destinations_device[:size],
            target_tokens_gpu[buffer.source_indices_device[:size]],
        )
        if buffer.completion_event is not None:
            buffer.completion_event.record(torch.cuda.current_stream(self.device))
            buffer.pending = True

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                self._free_req_resources(req_to_free)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        if self.speculator is not None:
            self.speculator.release(req)
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def _prepare_mappings(self, batch: Batch) -> Tuple[Indice2D, Indice2D]:
        buffer = self._mapping_staging_buffers[self._mapping_staging_buffer_index]
        self._mapping_staging_buffer_index = (self._mapping_staging_buffer_index + 1) % len(
            self._mapping_staging_buffers
        )
        if buffer.pending:
            assert buffer.completion_event is not None
            buffer.completion_event.synchronize()

        needed_size = batch.padded_forward_size
        assert needed_size <= len(buffer.positions_host)
        offset = 0
        for i, req in enumerate(batch.padded_reqs):
            length = batch.forward_extend_len(i)
            end = offset + length
            torch.arange(
                req.cached_len,
                req.cached_len + length,
                out=buffer.positions_host[offset:end],
            )
            buffer.req_mapping_host[offset:end].fill_(req.table_idx)
            offset = end
        assert offset == needed_size

        buffer.positions_device[:needed_size].copy_(
            buffer.positions_host[:needed_size], non_blocking=True
        )
        buffer.positions_index_device[:needed_size].copy_(
            buffer.positions_device[:needed_size], non_blocking=True
        )
        buffer.req_mapping_device[:needed_size].copy_(
            buffer.req_mapping_host[:needed_size], non_blocking=True
        )
        batch.positions = buffer.positions_device[:needed_size]
        input_mapping = (
            buffer.req_mapping_device[:needed_size],
            buffer.positions_index_device[:needed_size],
        )

        if batch.is_verify:
            write_mapping = (
                buffer.write_mapping_device[:0],
                buffer.write_positions_device[:0],
            )
        else:
            for i, req in enumerate(batch.reqs):
                buffer.write_mapping_host[i] = req.table_idx
                buffer.write_positions_host[i] = req.device_len if req.can_decode else -1
            size = batch.size
            buffer.write_mapping_device[:size].copy_(
                buffer.write_mapping_host[:size], non_blocking=True
            )
            buffer.write_positions_device[:size].copy_(
                buffer.write_positions_host[:size], non_blocking=True
            )
            write_mapping = (
                buffer.write_mapping_device[:size],
                buffer.write_positions_device[:size],
            )

        batch._mapping_staging_buffer = buffer
        buffer.pending = False
        return input_mapping, write_mapping

    def _mark_mapping_staging_consumed(self, batch: Batch) -> None:
        buffer = getattr(batch, "_mapping_staging_buffer", None)
        if buffer is None or buffer.completion_event is None:
            return
        buffer.completion_event.record(torch.cuda.current_stream(self.device))
        buffer.pending = True

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        with self._profile_region("minisgl_graph_pad"):
            self.engine.graph_runner.pad_batch(batch)
        with self._profile_region("minisgl_stage_drafts"):
            if batch.is_verify:
                self._stage_drafts(batch)
        with self._profile_region("minisgl_cache_allocate"):
            self.cache_manager.allocate_paged(batch)
            if batch.is_verify and batch.verify_width is not None:
                self._stage_verify_padding(batch)
        with self._profile_region("minisgl_prepare_mappings"):
            input_mapping, write_mapping = self._prepare_mappings(batch)
            batch.out_loc = self.engine.page_table[input_mapping]
        with self._profile_region("minisgl_attention_metadata"):
            self.engine.attn_backend.prepare_metadata(batch)
        with self._profile_region("minisgl_sampling_prepare"):
            sample_args = (
                self.speculator.prepare_sampling(batch, self.engine.sampler)
                if batch.is_verify and self.speculator is not None
                else self.engine.sampler.prepare(batch)
            )
        return ForwardInput(
            batch=batch,
            sample_args=sample_args,
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _schedule_next_batch(self, inflight_reqs: Set[Req] | None = None) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        with self._profile_region("minisgl_prefill_schedule"):
            batch = self.prefill_manager.schedule_next_batch(self.prefill_budget)
        if batch is None:
            if self.speculator is not None:
                decode_reqs = self.decode_manager.running_reqs
                if self.speculative_overlap_enabled:
                    decode_reqs = decode_reqs.difference(inflight_reqs or set())
                    ordered_reqs = sorted(decode_reqs, key=lambda req: req.uid)
                    if self.speculative_overlap_batch_size is not None:
                        ordered_reqs = ordered_reqs[: self.speculative_overlap_batch_size]
                    decode_reqs = set(ordered_reqs)
                with self._profile_region("minisgl_ngram_draft_schedule"):
                    batch = self.speculator.schedule(decode_reqs)
            else:
                with self._profile_region("minisgl_decode_schedule"):
                    batch = self.decode_manager.schedule_next_batch()
        return self._prepare_batch(batch) if batch else None

    def _stage_drafts(self, batch: Batch) -> None:
        assert batch.draft_ids is not None
        assert self._draft_staging_buffers
        buffer = self._draft_staging_buffers[self._draft_staging_buffer_index]
        self._draft_staging_buffer_index = (self._draft_staging_buffer_index + 1) % len(
            self._draft_staging_buffers
        )
        if buffer.pending:
            assert buffer.completion_event is not None
            buffer.completion_event.synchronize()

        offset = 0
        row_width = self.token_pool.size(1)
        for i, (req, draft_ids) in enumerate(zip(batch.reqs, batch.draft_ids, strict=True)):
            start = req.device_len
            draft_end = start + len(draft_ids)
            assert draft_end < req.max_device_len
            physical_end = req.cached_len + batch.forward_extend_len(i)
            assert physical_end <= self.engine.max_seq_len
            staged_len = physical_end - start
            if staged_len == 0:
                continue

            staged_end = offset + staged_len
            assert staged_end <= len(buffer.token_ids_host)
            real_draft_end = offset + len(draft_ids)
            buffer.token_ids_host[offset:real_draft_end].copy_(draft_ids)
            if real_draft_end < staged_end:
                buffer.token_ids_host[real_draft_end:staged_end].zero_()
            flat_start = req.table_idx * row_width + start
            torch.arange(
                flat_start,
                flat_start + staged_len,
                out=buffer.locations_host[offset:staged_end],
            )
            offset = staged_end

        if offset == 0:
            buffer.pending = False
            return

        buffer.token_ids_device[:offset].copy_(buffer.token_ids_host[:offset], non_blocking=True)
        buffer.locations_device[:offset].copy_(buffer.locations_host[:offset], non_blocking=True)
        self.token_pool.view(-1).index_copy_(
            0,
            buffer.locations_device[:offset],
            buffer.token_ids_device[:offset],
        )
        if buffer.completion_event is not None:
            buffer.completion_event.record(torch.cuda.current_stream(self.device))
            buffer.pending = True

    def _stage_verify_padding(self, batch: Batch) -> None:
        """Map ignored graph rows to the shared dummy KV page."""
        assert batch.is_verify and batch.verify_width is not None
        dummy_table = self.engine.page_table[self.engine.dummy_req.table_idx]
        for i, req in enumerate(batch.reqs):
            padding_start = batch.allocated_device_len(i)
            padding_end = batch.forward_device_len(i)
            padding_len = padding_end - padding_start
            if padding_len > 0:
                self.engine.page_table[req.table_idx, padding_start:padding_end].copy_(
                    dummy_table[:padding_len]
                )

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        if batch.is_verify:
            assert self.speculator is not None

            def token_selector(logits: torch.Tensor) -> torch.Tensor:
                assert self.speculator is not None
                return self.speculator.select_verification_tokens(
                    batch,
                    logits,
                    self.engine.sampler,
                    sample_args,
                )

        else:
            token_selector = None
        if self.profiler is None:
            forward_output = self.engine.forward_batch(
                batch,
                sample_args,
                token_selector=token_selector,
            )
        else:
            label = f"minisgl_forward_{batch.phase}_bs{batch.size}_rows{batch.forward_size}"
            with torch.profiler.record_function(label):
                forward_output = self.engine.forward_batch(
                    batch,
                    sample_args,
                    token_selector=token_selector,
                )
            self.profiler.step()
        if not batch.is_verify:
            self.token_pool[output_mapping] = forward_output.next_tokens_gpu
            self.decode_manager.filter_reqs(forward_input.batch.reqs)
        self._mark_mapping_staging_consumed(batch)
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(batch.forward_extend_len(i) for i in range(batch.padded_size))
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for i, req in enumerate(batch.padded_reqs):
        length = batch.forward_extend_len(i)
        torch.arange(
            req.cached_len,
            req.cached_len + length,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for i, req in enumerate(batch.padded_reqs):
        length = batch.forward_extend_len(i)
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
