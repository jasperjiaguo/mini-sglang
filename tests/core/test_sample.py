from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

import minisgl.engine.sample as sample_module
from minisgl.core import SamplingParams
from minisgl.engine.sample import BatchSamplingArgs, Sampler, apply_greedy_mask


def test_prepare_params_marks_greedy_rows_in_mixed_batch(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        sample_module,
        "make_device_tensor",
        lambda data, dtype, device: torch.tensor(data, dtype=dtype),
    )
    sampler = Sampler(device=torch.device("cpu"), vocab_size=10)

    args = sampler.prepare_params(
        [
            SamplingParams(temperature=0.0),
            SamplingParams(temperature=0.7, top_k=8, top_p=0.9),
        ]
    )

    assert args.temperatures is not None
    assert args.temperatures.tolist() == pytest.approx([1.0, 0.7])
    assert args.top_k is not None and args.top_k.tolist() == [10, 8]
    assert args.top_p is not None and args.top_p.tolist() == pytest.approx([1.0, 0.9])
    assert args.greedy_mask is not None and args.greedy_mask.tolist() == [True, False]


def test_prepare_params_uses_fast_path_for_all_greedy_requests():
    sampler = Sampler(device=torch.device("cpu"), vocab_size=10)

    args = sampler.prepare_params(
        [
            SamplingParams(temperature=0.0),
            SamplingParams(temperature=0.7, top_k=1),
        ]
    )

    assert args.temperatures is None
    assert args.greedy_mask is None


def test_sample_uses_argmax_when_temperature_is_not_set(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda name: nullcontext())
    sampler = Sampler(device=torch.device("cpu"), vocab_size=3)
    logits = torch.tensor(
        [
            [1.0, 3.0, 2.0],
            [4.0, 2.0, 1.0],
        ]
    )

    result = sampler.sample(logits, BatchSamplingArgs(temperatures=None))

    assert result.tolist() == [1, 0]


def test_apply_greedy_mask_preserves_mixed_request_semantics():
    logits = torch.tensor(
        [
            [1.0, 3.0, 2.0],
            [4.0, 2.0, 1.0],
        ]
    )
    sampled = torch.tensor([0, 2])
    greedy_mask = torch.tensor([True, False])

    result = apply_greedy_mask(logits, sampled, greedy_mask)

    assert result.tolist() == [1, 2]
