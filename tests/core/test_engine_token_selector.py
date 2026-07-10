from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import Mock

import pytest
import torch

from minisgl.core import Batch, Req, SamplingParams
from minisgl.engine.engine import Engine
from minisgl.engine.sample import BatchSamplingArgs


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="engine token-selection tests require CUDA",
)


class _FakeContext:
    @contextmanager
    def forward_batch(self, batch: Batch):
        yield


class _FakeGraphRunner:
    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return False


def _make_req(phase: str) -> tuple[Req, Batch]:
    req = Req(
        input_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
        table_idx=0,
        cached_len=2,
        output_len=8,
        uid=0,
        sampling_params=SamplingParams(),
        cache_handle=None,  # type: ignore[arg-type]
    )
    batch = Batch(
        reqs=[req],
        phase=phase,  # type: ignore[arg-type]
        draft_ids=(
            [torch.tensor([4], dtype=torch.int32)] if phase == "verify" else None
        ),
    )
    return req, batch


def _make_engine(logits: torch.Tensor, sampler: Mock) -> Engine:
    engine = object.__new__(Engine)
    engine.stream = torch.cuda.current_stream()
    engine.ctx = _FakeContext()
    engine.graph_runner = _FakeGraphRunner()
    engine.model = Mock()
    engine.model.forward.return_value = logits
    engine.sampler = sampler
    return engine


def test_custom_token_selector_receives_raw_verification_logits():
    req, batch = _make_req("verify")
    logits = torch.randn(2, 16, device="cuda")
    sampler = Mock()
    engine = _make_engine(logits, sampler)
    selected = torch.tensor([5, 6], dtype=torch.int64, device="cuda")
    token_selector = Mock(return_value=selected)
    original_state = req.cached_len, req.device_len

    output = engine.forward_batch(
        batch,
        BatchSamplingArgs(temperatures=None),
        token_selector=token_selector,
    )
    output.copy_done_event.synchronize()

    assert token_selector.call_args.args[0] is logits
    sampler.sample.assert_not_called()
    assert output.next_tokens_gpu.dtype == torch.int32
    assert output.next_tokens_cpu.tolist() == [5, 6]
    assert (req.cached_len, req.device_len) == original_state


def test_default_token_selector_preserves_normal_decode_completion():
    req, batch = _make_req("decode")
    logits = torch.randn(1, 16, device="cuda")
    sampler = Mock()
    sampler.sample.return_value = torch.tensor([7], dtype=torch.int64, device="cuda")
    engine = _make_engine(logits, sampler)
    old_device_len = req.device_len
    args = BatchSamplingArgs(temperatures=None)

    output = engine.forward_batch(batch, args)
    output.copy_done_event.synchronize()

    sampled_logits, sampled_args = sampler.sample.call_args.args
    assert torch.equal(sampled_logits, logits)
    assert sampled_args is args
    assert output.next_tokens_cpu.tolist() == [7]
    assert req.cached_len == old_device_len
    assert req.device_len == old_device_len + 1
