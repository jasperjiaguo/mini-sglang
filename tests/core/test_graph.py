from __future__ import annotations

import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.engine.graph import GraphRunner


def _make_req(uid: int, *, input_len: int = 5, output_len: int = 8) -> Req:
    return Req(
        input_ids=torch.arange(input_len, dtype=torch.int32),
        table_idx=uid,
        cached_len=input_len - 1,
        output_len=output_len,
        uid=uid,
        sampling_params=SamplingParams(),
        cache_handle=None,  # type: ignore[arg-type]
    )


def _make_runner(*, max_bs: int = 4, verify_width: int | None = 4) -> GraphRunner:
    runner = object.__new__(GraphRunner)
    runner.max_graph_bs = max_bs
    runner.graph_bs_list = [1, 2, 4][:max_bs]
    runner.verify_width = verify_width
    runner.logical_max_seq_len = 32
    runner.dummy_req = _make_req(-1, input_len=1, output_len=1)
    return runner


def test_decode_graph_batch_padding_is_unchanged():
    runner = _make_runner()
    batch = Batch(reqs=[_make_req(i) for i in range(3)], phase="decode")

    runner.pad_batch(batch)

    assert batch.verify_width is None
    assert batch.size == 3
    assert batch.padded_size == 4
    assert batch.forward_size == 3
    assert batch.padded_forward_size == 4


def test_verification_graph_pads_short_ngram_continuation_to_fixed_width():
    runner = _make_runner(verify_width=4)
    req = _make_req(0)
    batch = Batch(
        reqs=[req],
        phase="verify",
        draft_ids=[[7]],
    )

    runner.pad_batch(batch)

    assert batch.verify_width == 4
    assert batch.verification_len(0) == 2
    assert batch.forward_extend_len(0) == 4
    assert batch.forward_device_len(0) == req.cached_len + 4
    assert batch.allocated_device_len(0) == req.cached_len + 2


def test_verification_graph_can_pad_when_only_two_output_slots_remain():
    runner = _make_runner(verify_width=4)
    req = _make_req(0)
    req.max_device_len = req.device_len + 2
    batch = Batch(
        reqs=[req],
        phase="verify",
        draft_ids=[[7]],
    )

    runner.pad_batch(batch)

    assert req.remain_len == 2
    assert batch.verify_width == 4
    assert batch.verification_len(0) == req.remain_len


def test_verification_graph_falls_back_when_padding_exceeds_model_context():
    runner = _make_runner(verify_width=4)
    runner.logical_max_seq_len = 6
    req = _make_req(0)
    batch = Batch(
        reqs=[req],
        phase="verify",
        draft_ids=[[7]],
    )

    runner.pad_batch(batch)

    assert batch.verify_width is None
    assert batch.padded_reqs == batch.reqs
    assert batch.forward_extend_len(0) == 2


def test_disabled_cuda_graph_keeps_variable_verification_width():
    runner = _make_runner(max_bs=0, verify_width=4)
    req = _make_req(0)
    batch = Batch(
        reqs=[req],
        phase="verify",
        draft_ids=[[7]],
    )

    runner.pad_batch(batch)

    assert batch.verify_width is None
    assert batch.forward_extend_len(0) == 2
    assert batch.forward_device_len(0) == req.device_len + 1
