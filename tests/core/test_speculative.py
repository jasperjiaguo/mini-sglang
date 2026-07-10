from __future__ import annotations

from typing import Any, Literal
from unittest.mock import Mock

import pytest
import torch

from minisgl.core import Batch, Req, SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.env import ENV
from minisgl.scheduler.config import SchedulerConfig
from minisgl.scheduler.scheduler import Scheduler
from minisgl.scheduler.speculative import (
    NgramSpeculator,
    SpeculativeStats,
    _create_speculator,
    find_ngram_draft,
    greedy_accept,
)


def test_find_ngram_draft_uses_most_recent_match():
    ids = torch.tensor([1, 2, 3, 1, 2, 4, 1, 2], dtype=torch.int32)
    draft = find_ngram_draft(ids, ngram_size=2, max_draft_tokens=3)
    assert draft.tolist() == [4, 1, 2]


def test_find_ngram_draft_returns_empty_without_match():
    ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    draft = find_ngram_draft(ids, ngram_size=2, max_draft_tokens=3)
    assert draft.tolist() == []


def test_find_ngram_draft_allows_overlapping_match():
    ids = torch.tensor([7, 7, 7], dtype=torch.int32)
    draft = find_ngram_draft(ids, ngram_size=1, max_draft_tokens=3)
    assert draft.tolist() == [7]


def test_greedy_accept_stops_at_first_mismatch():
    result = greedy_accept(
        torch.tensor([3, 4, 5], dtype=torch.int32),
        torch.tensor([3, 9, 8, 7], dtype=torch.int32),
    )
    assert result.token_ids.tolist() == [3, 9]
    assert result.accepted_drafts == 1


def test_greedy_accept_returns_bonus_when_all_drafts_match():
    result = greedy_accept(
        torch.tensor([3, 4, 5], dtype=torch.int32),
        torch.tensor([3, 4, 5, 9], dtype=torch.int32),
    )
    assert result.token_ids.tolist() == [3, 4, 5, 9]
    assert result.accepted_drafts == 3


def test_verify_batch_extends_the_pending_token_and_drafts():
    req = _make_req(0, [1, 2, 3, 1, 2])
    batch = Batch(
        reqs=[req],
        phase="verify",
        draft_ids=[torch.tensor([3, 1, 2], dtype=torch.int32)],
    )
    batch.padded_reqs = batch.reqs

    assert req.extend_len == 1
    assert batch.forward_extend_len(0) == 4
    assert batch.forward_device_len(0) == len(req.input_ids) + 3


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_non_speculative_batch_lengths_match_request_state(
    phase: Literal["prefill", "decode"],
):
    reqs = [_make_req(0, [1, 2, 3, 4, 5]), _make_req(1, [6, 7, 8])]
    reqs[0].cached_len = 2
    reqs[1].cached_len = 1
    batch = Batch(reqs=reqs, phase=phase)
    batch.padded_reqs = batch.reqs

    for i, req in enumerate(reqs):
        assert batch.forward_extend_len(i) == req.extend_len
        assert batch.forward_device_len(i) == req.device_len


def _make_req(uid: int, ids: list[int], *, greedy: bool = True) -> Req:
    return Req(
        input_ids=torch.tensor(ids, dtype=torch.int32),
        table_idx=uid,
        cached_len=len(ids) - 1,
        output_len=8,
        uid=uid,
        sampling_params=SamplingParams(temperature=0.0 if greedy else 0.7),
        cache_handle=None,  # type: ignore[arg-type]
    )


def _spec_config(
    algorithm: Literal["ngram"] | None = None,
    raw_config: str | None = None,
) -> SchedulerConfig:
    return SchedulerConfig(
        model_path="unused",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        spec_decoding=algorithm,
        spec_decoding_config=raw_config,
        page_size=1,
        attention_backend="fa",
    )


def test_speculative_decoding_is_off_without_an_algorithm():
    config = SchedulerConfig(
        model_path="unused",
        tp_info=DistributedInfo(0, 2),
        dtype=torch.bfloat16,
        page_size=16,
        attention_backend="trtllm",
    )
    assert config.spec_decoding is None
    assert config.spec_decoding_config is None
    assert _create_speculator(config) is None


def test_speculative_config_requires_an_algorithm():
    with pytest.raises(ValueError, match="requires --spec-decoding"):
        _create_speculator(
            _spec_config(raw_config='{"ngram_size": 2, "num_draft_tokens": 3}')
        )


def test_ngram_algorithm_requires_a_config():
    with pytest.raises(ValueError, match="requires --spec-decoding-config"):
        _create_speculator(_spec_config(algorithm="ngram"))


@pytest.mark.parametrize(
    "raw_config",
    [
        "not-json",
        "[]",
        '{"ngram_size": 2}',
        '{"ngram_size": 2, "num_draft_tokens": 3, "extra": 4}',
        '{"ngram_size": 0, "num_draft_tokens": 3}',
        '{"ngram_size": 2, "num_draft_tokens": true}',
    ],
)
def test_ngram_config_rejects_invalid_json_values(raw_config: str):
    with pytest.raises(ValueError):
        _create_speculator(_spec_config(algorithm="ngram", raw_config=raw_config))


def test_ngram_config_constructs_speculator(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ENV.DISABLE_OVERLAP_SCHEDULING, "value", True)
    speculator = _create_speculator(
        _spec_config(
            algorithm="ngram",
            raw_config='{"ngram_size": 2, "num_draft_tokens": 3}',
        )
    )

    assert isinstance(speculator, NgramSpeculator)
    assert speculator.ngram_size == 2
    assert speculator.num_draft_tokens == 3


def test_cli_parses_explicit_ngram_configuration():
    from minisgl.server.args import parse_args

    config, _ = parse_args(
        [
            "--model",
            "unused",
            "--dtype",
            "bfloat16",
            "--spec-decoding",
            "ngram",
            "--spec-decoding-config",
            '{"ngram_size": 2, "num_draft_tokens": 3}',
        ]
    )

    assert config.spec_decoding == "ngram"
    assert config.spec_decoding_config == '{"ngram_size": 2, "num_draft_tokens": 3}'


@pytest.mark.parametrize(
    "spec_args",
    [
        ["--spec-decoding-config", '{"ngram_size": 2, "num_draft_tokens": 3}'],
        ["--spec-decoding", "ngram"],
    ],
)
def test_cli_requires_algorithm_and_config_together(spec_args: list[str]):
    from minisgl.server.args import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--model", "unused", "--dtype", "bfloat16", *spec_args])


def test_scheduler_uses_normal_decode_path_when_speculation_is_off():
    req = _make_req(0, [1, 2, 3])
    decode_batch = Batch(reqs=[req], phase="decode")
    scheduler: Any = object.__new__(Scheduler)
    scheduler.prefill_budget = 128
    scheduler.speculator = None
    scheduler.prefill_manager = Mock()
    scheduler.prefill_manager.schedule_next_batch.return_value = None
    scheduler.decode_manager = Mock()
    scheduler.decode_manager.schedule_next_batch.return_value = decode_batch
    scheduler._prepare_batch = lambda batch: batch

    result = scheduler._schedule_next_batch()

    assert result is decode_batch
    scheduler.prefill_manager.schedule_next_batch.assert_called_once_with(128)
    scheduler.decode_manager.schedule_next_batch.assert_called_once_with()


def test_scheduler_preserves_prefill_priority_when_speculation_is_off():
    req = _make_req(0, [1, 2, 3])
    prefill_batch = Batch(reqs=[req], phase="prefill")
    scheduler: Any = object.__new__(Scheduler)
    scheduler.prefill_budget = 128
    scheduler.speculator = None
    scheduler.prefill_manager = Mock()
    scheduler.prefill_manager.schedule_next_batch.return_value = prefill_batch
    scheduler.decode_manager = Mock()
    scheduler._prepare_batch = lambda batch: batch

    result = scheduler._schedule_next_batch()

    assert result is prefill_batch
    scheduler.prefill_manager.schedule_next_batch.assert_called_once_with(128)
    scheduler.decode_manager.schedule_next_batch.assert_not_called()


def test_speculator_separates_verify_and_normal_requests_fairly():
    speculator = NgramSpeculator(ngram_size=2, num_draft_tokens=3)
    verify_req = _make_req(0, [1, 2, 3, 1, 2])
    sampled_req = _make_req(1, [1, 2, 3], greedy=False)

    first = speculator.schedule([sampled_req, verify_req])
    second = speculator.schedule([sampled_req, verify_req])

    assert first is not None and first.is_verify
    assert first.reqs == [verify_req]
    assert first.draft_ids is not None and first.draft_ids[0].tolist() == [3, 1, 2]
    assert second is not None and second.is_decode
    assert second.reqs == [sampled_req]


def test_speculator_reserves_one_output_token_for_the_bonus_token():
    speculator = NgramSpeculator(ngram_size=2, num_draft_tokens=3)
    req = _make_req(0, [1, 2, 3, 1, 2])
    req.max_device_len = req.device_len + 2

    batch = speculator.schedule([req])

    assert batch is not None and batch.is_verify
    assert batch.draft_ids is not None and batch.draft_ids[0].tolist() == [3]


def test_speculator_owns_verification_and_metrics():
    speculator = NgramSpeculator(ngram_size=2, num_draft_tokens=3)
    req = _make_req(0, [1, 2, 3, 1, 2])
    batch = speculator.schedule([req])
    assert batch is not None and batch.is_verify

    acceptance = speculator.verify(
        batch,
        0,
        torch.tensor([3, 9, 8, 7], dtype=torch.int32),
    )
    speculator.record_verification(batch, 0, acceptance.accepted_drafts)

    assert acceptance.token_ids.tolist() == [3, 9]
    assert speculator.stats.verify_steps == 1
    assert speculator.stats.drafted_tokens == 3
    assert speculator.stats.accepted_drafts == 1


def test_speculative_stats_track_lookup_failures_and_position_acceptance():
    stats = SpeculativeStats()
    stats.record_lookup(matched=True)
    stats.record_lookup(matched=False)
    stats.record_lookup(matched=False)
    stats.record_verify(drafted_tokens=3, accepted_drafts=3)
    stats.record_verify(drafted_tokens=3, accepted_drafts=1)
    stats.record_verify(drafted_tokens=2, accepted_drafts=0)

    assert stats.lookup_attempts == 3
    assert stats.lookup_matches == 1
    assert stats.lookup_misses == 2
    assert stats.lookup_match_rate == 1 / 3
    assert stats.position_attempts == [3, 2, 1]
    assert stats.position_accepts == [2, 1, 1]
