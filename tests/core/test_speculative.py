from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import Mock

import pytest
import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.engine.sample import BatchSamplingArgs
from minisgl.env import ENV
from minisgl.scheduler.config import SchedulerConfig
from minisgl.scheduler.scheduler import (
    Scheduler,
    _DraftStagingBuffer,
    _MappingStagingBuffer,
    _PendingTokenStagingBuffer,
)
from minisgl.scheduler.speculative import (
    NgramSpeculator,
    SpeculativeStats,
    _create_speculator,
    _find_ngram_draft,
    accept_deterministic_draft,
    find_ngram_draft,
)


def test_find_ngram_draft_uses_most_recent_match():
    ids = torch.tensor([1, 2, 3, 1, 2, 4, 1, 2], dtype=torch.int32)
    draft = find_ngram_draft(ids, ngram_size=2, max_draft_tokens=3)
    assert draft.tolist() == [4, 1, 2]


def test_find_ngram_draft_prefers_longest_suffix_match():
    ids = torch.tensor([1, 2, 8, 2, 9, 1, 2], dtype=torch.int32)
    draft = find_ngram_draft(ids, ngram_size=2, max_draft_tokens=2)
    assert draft.tolist() == [8, 2]


def test_find_ngram_draft_falls_back_to_shorter_suffix():
    ids = torch.tensor([4, 2, 8, 3, 2, 9, 1, 2], dtype=torch.int32)
    draft = find_ngram_draft(ids, ngram_size=3, max_draft_tokens=3)
    assert draft.tolist() == [9, 1, 2]


def test_find_ngram_draft_falls_back_when_ngram_exceeds_history():
    ids = torch.tensor([5, 6, 5], dtype=torch.int32)
    draft = find_ngram_draft(ids, ngram_size=8, max_draft_tokens=2)
    assert draft.tolist() == [6, 5]


def test_find_ngram_draft_returns_empty_without_match():
    ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    draft = find_ngram_draft(ids, ngram_size=2, max_draft_tokens=3)
    assert draft.tolist() == []


def test_find_ngram_draft_allows_overlapping_match():
    ids = torch.tensor([7, 7, 7], dtype=torch.int32)
    draft = find_ngram_draft(ids, ngram_size=1, max_draft_tokens=3)
    assert draft.tolist() == [7]


def test_incremental_ngram_index_matches_full_lookup_after_appends() -> None:
    speculator = NgramSpeculator(ngram_size=3, num_draft_tokens=3)
    req = _make_req(0, [4, 2, 8, 3, 2, 9, 1, 2])
    req.max_device_len = req.device_len + 16

    for token in [9, 1, 2, 7, 2]:
        expected, expected_size = _find_ngram_draft(req.input_ids, 3, 3)
        actual = speculator._draft(req)
        assert actual == expected.tolist()
        if expected_size:
            assert speculator.stats.lookup_matches_by_size.get(expected_size, 0) >= 1
        req.append_host(torch.tensor([token], dtype=torch.int32))
        req.device_len += 1

    assert req in speculator._history_indices
    speculator.release(req)
    assert req not in speculator._history_indices


def test_verification_updates_ngram_history_without_resync() -> None:
    speculator = NgramSpeculator(ngram_size=3, num_draft_tokens=2)
    req = _make_req(0, [1, 2, 3, 1, 2])
    speculator._draft(req)
    req.append_host(torch.tensor([3, 4], dtype=torch.int32))

    speculator.update_history(req, [3, 4])

    index = speculator._history_indices[req]
    assert index.tokens == req.input_ids.tolist()


def test_request_append_reuses_preallocated_history_storage() -> None:
    req = _make_req(0, [1, 2, 3])
    storage_ptr = req._input_ids_storage.data_ptr()

    req.append_host(torch.tensor([4, 5], dtype=torch.int32))

    assert req.input_ids.tolist() == [1, 2, 3, 4, 5]
    assert req.input_ids.data_ptr() == storage_ptr


def test_deterministic_rejection_stops_at_first_mismatch():
    result = accept_deterministic_draft(
        torch.tensor([3, 4, 5], dtype=torch.int32),
        torch.tensor([3, 9, 8, 7], dtype=torch.int32),
    )
    assert result.token_ids.tolist() == [3, 9]
    assert result.accepted_drafts == 1


def test_deterministic_rejection_can_reject_first_draft():
    result = accept_deterministic_draft(
        torch.tensor([3, 4, 5], dtype=torch.int32),
        torch.tensor([9, 8, 7, 6], dtype=torch.int32),
    )
    assert result.token_ids.tolist() == [9]
    assert result.accepted_drafts == 0


def test_deterministic_rejection_returns_bonus_when_all_drafts_match():
    result = accept_deterministic_draft(
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
        draft_ids=[[3, 1, 2]],
    )
    batch.padded_reqs = batch.reqs

    assert req.extend_len == 1
    assert batch.forward_extend_len(0) == 4
    assert batch.forward_device_len(0) == len(req.input_ids) + 3


def test_verify_batch_separates_real_drafts_from_graph_execution_width():
    req = _make_req(0, [1, 2, 3, 1, 2])
    batch = Batch(
        reqs=[req],
        phase="verify",
        draft_ids=[[3]],
        verify_width=4,
    )
    batch.padded_reqs = batch.reqs

    assert batch.verification_len(0) == 2
    assert batch.forward_extend_len(0) == 4
    assert batch.forward_device_len(0) == req.cached_len + 4


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
        _create_speculator(_spec_config(raw_config='{"ngram_size": 2, "num_draft_tokens": 3}'))


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
    assert speculator.cuda_graph_verify_width == 4


def test_ngram_config_allows_overlap_scheduling(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ENV.DISABLE_OVERLAP_SCHEDULING, "value", False)

    speculator = _create_speculator(
        _spec_config(
            algorithm="ngram",
            raw_config='{"ngram_size": 2, "num_draft_tokens": 3}',
        )
    )

    assert isinstance(speculator, NgramSpeculator)


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


def test_overlap_scheduler_uses_a_disjoint_capped_speculative_batch():
    reqs = [_make_req(uid, [uid + 1, 9, uid + 1, 9]) for uid in range(4)]
    scheduler: Any = object.__new__(Scheduler)
    scheduler.prefill_budget = 128
    scheduler.speculator = NgramSpeculator(ngram_size=2, num_draft_tokens=1)
    scheduler.speculative_overlap_enabled = True
    scheduler.speculative_overlap_batch_size = 2
    scheduler.prefill_manager = Mock()
    scheduler.prefill_manager.schedule_next_batch.return_value = None
    scheduler.decode_manager = Mock()
    scheduler.decode_manager.running_reqs = set(reqs)
    scheduler._prepare_batch = lambda batch: batch

    batch = scheduler._schedule_next_batch(set(reqs[:2]))

    assert batch is not None
    assert batch.reqs == reqs[2:]
    assert set(batch.reqs).isdisjoint(reqs[:2])


def test_speculator_alternates_verify_and_normal_requests_fairly():
    speculator = NgramSpeculator(ngram_size=2, num_draft_tokens=3)
    verify_req = _make_req(0, [1, 2, 3, 1, 2])
    no_draft_req = _make_req(1, [1, 2, 3], greedy=False)

    first = speculator.schedule([no_draft_req, verify_req])
    second = speculator.schedule([no_draft_req, verify_req])

    assert first is not None and first.is_verify
    assert first.reqs == [verify_req]
    assert first.draft_ids is not None and first.draft_ids[0] == [3, 1, 2]
    assert second is not None and second.is_decode
    assert second.reqs == [no_draft_req]


def test_speculator_drafts_for_mixed_greedy_and_sampled_requests():
    speculator = NgramSpeculator(ngram_size=2, num_draft_tokens=3)
    greedy_req = _make_req(0, [1, 2, 3, 1, 2])
    sampled_req = _make_req(1, [4, 5, 6, 4, 5], greedy=False)

    batch = speculator.schedule([sampled_req, greedy_req])

    assert batch is not None and batch.is_verify
    assert batch.reqs == [greedy_req, sampled_req]
    assert batch.draft_ids is not None
    assert batch.draft_ids == [[3, 1, 2], [6, 4, 5]]


def test_speculator_keeps_misses_in_mixed_verification_batch():
    speculator = NgramSpeculator(
        ngram_size=2,
        num_draft_tokens=3,
        mixed_batch=True,
    )
    verify_req = _make_req(0, [1, 2, 3, 1, 2])
    no_draft_req = _make_req(1, [4, 5, 6])

    batch = speculator.schedule([no_draft_req, verify_req])

    assert batch is not None and batch.is_verify
    assert batch.reqs == [verify_req, no_draft_req]
    assert batch.draft_ids is not None
    assert batch.draft_ids == [[3, 1, 2], []]
    assert speculator.stats.verify_batches == 1
    assert speculator.stats.decode_batches == 0
    assert speculator.stats.folded_decode_rows == 1


def test_speculator_does_not_count_empty_mixed_drafts_as_verification() -> None:
    speculator = NgramSpeculator(
        ngram_size=2,
        num_draft_tokens=3,
        mixed_batch=True,
    )
    req = _make_req(0, [1, 2, 3])
    batch = Batch(
        reqs=[req],
        phase="verify",
        draft_ids=[[]],
    )

    speculator.record_verification(batch, 0, accepted_drafts=0)

    assert speculator.stats.verify_steps == 0


def test_mixed_speculator_folds_an_all_miss_batch_into_verification() -> None:
    speculator = NgramSpeculator(
        ngram_size=2,
        num_draft_tokens=3,
        mixed_batch=True,
    )
    reqs = [_make_req(0, [1, 2, 3]), _make_req(1, [4, 5, 6])]

    batch = speculator.schedule(reqs)

    assert batch is not None and batch.is_verify
    assert batch.draft_ids is not None
    assert batch.draft_ids == [[], []]
    assert speculator.stats.decode_batches == 0
    assert speculator.stats.folded_decode_rows == 2


def test_speculator_records_fallback_match_size():
    speculator = NgramSpeculator(ngram_size=3, num_draft_tokens=3)
    req = _make_req(0, [4, 2, 8, 3, 2, 9, 1, 2])

    batch = speculator.schedule([req])

    assert batch is not None and batch.is_verify
    assert batch.draft_ids is not None and batch.draft_ids[0] == [9, 1, 2]
    assert speculator.stats.lookup_matches_by_size == {1: 1}


def test_speculator_repeats_sampling_params_for_each_verification_row():
    speculator = NgramSpeculator(ngram_size=2, num_draft_tokens=3)
    greedy_req = _make_req(0, [1, 2, 3])
    sampled_req = _make_req(1, [4, 5, 6], greedy=False)
    batch = Batch(
        reqs=[greedy_req, sampled_req],
        phase="verify",
        draft_ids=[[7, 8], [9]],
    )
    batch.padded_reqs = batch.reqs
    sampler = Mock()
    sentinel = BatchSamplingArgs(temperatures=None)
    sampler.prepare_params.return_value = sentinel

    result = speculator.prepare_sampling(batch, sampler)

    assert result is sentinel
    sampler.prepare_params.assert_called_once_with(
        [
            greedy_req.sampling_params,
            greedy_req.sampling_params,
            greedy_req.sampling_params,
            sampled_req.sampling_params,
            sampled_req.sampling_params,
        ]
    )


def test_speculator_prepares_sampling_for_padded_graph_rows():
    speculator = NgramSpeculator(ngram_size=2, num_draft_tokens=3)
    req = _make_req(0, [1, 2, 3])
    batch = Batch(
        reqs=[req],
        phase="verify",
        draft_ids=[[7]],
        verify_width=4,
    )
    batch.padded_reqs = batch.reqs
    sampler = Mock()
    sentinel = BatchSamplingArgs(temperatures=None)
    sampler.prepare_params.return_value = sentinel

    result = speculator.prepare_sampling(batch, sampler)

    assert result is sentinel
    sampler.prepare_params.assert_called_once_with([req.sampling_params])


def test_speculator_samples_verification_rows_before_rejection():
    speculator = NgramSpeculator(ngram_size=2, num_draft_tokens=2)
    req = _make_req(0, [1, 2, 3])
    batch = Batch(
        reqs=[req],
        phase="verify",
        draft_ids=[[3, 4]],
    )
    logits = torch.randn(3, 10)
    args = BatchSamplingArgs(temperatures=torch.ones(3))
    target_samples = torch.tensor([3, 9, 7], dtype=torch.int32)
    sampler = Mock()
    sampler.sample.return_value = target_samples

    predictions = speculator.select_verification_tokens(batch, logits, sampler, args)
    acceptance = speculator.verify(batch, 0, predictions)

    assert predictions is target_samples
    sampler.sample.assert_called_once_with(logits, args)
    assert acceptance.token_ids.tolist() == [3, 9]
    assert acceptance.accepted_drafts == 1


def test_speculator_ignores_padded_graph_targets_during_verification():
    speculator = NgramSpeculator(ngram_size=2, num_draft_tokens=3)
    req = _make_req(0, [1, 2, 3])
    batch = Batch(
        reqs=[req],
        phase="verify",
        draft_ids=[[3]],
        verify_width=4,
    )
    batch.padded_reqs = batch.reqs

    acceptance = speculator.verify(
        batch,
        0,
        torch.tensor([3, 9, 100, 101], dtype=torch.int32),
    )

    assert acceptance.token_ids.tolist() == [3, 9]
    assert acceptance.accepted_drafts == 1


def test_speculator_verifies_mixed_draft_lengths_in_one_batch():
    speculator = NgramSpeculator(ngram_size=2, num_draft_tokens=2)
    reqs = [
        _make_req(0, [1, 2, 3]),
        _make_req(1, [4, 5, 6]),
        _make_req(2, [7, 8, 9]),
    ]
    batch = Batch(
        reqs=reqs,
        phase="verify",
        draft_ids=[[7, 8], [4], []],
        verify_width=3,
    )
    batch.padded_reqs = batch.reqs
    target_tokens = torch.tensor([7, 9, 99, 8, 99, 99, 6, 99, 99], dtype=torch.int32)

    results = speculator.verify_batch(batch, target_tokens)

    assert [result.token_ids.tolist() for result in results] == [[7, 9], [8], [6]]
    assert [result.accepted_drafts for result in results] == [1, 0, 0]


def test_scheduler_stages_real_drafts_then_zero_padding():
    req = _make_req(0, [1, 2, 3, 1, 2])
    batch = Batch(
        reqs=[req],
        phase="verify",
        draft_ids=[[3]],
        verify_width=4,
    )
    batch.padded_reqs = batch.reqs
    scheduler: Any = object.__new__(Scheduler)
    scheduler.engine = SimpleNamespace(max_seq_len=16)
    scheduler.token_pool = torch.full((1, 16), 99, dtype=torch.int32)
    scheduler.device = torch.device("cpu")
    scheduler._draft_staging_buffer_index = 0
    scheduler._draft_staging_buffers = [
        _DraftStagingBuffer.create(3, scheduler.device, pin_memory=False)
    ]

    scheduler._stage_drafts(batch)

    assert scheduler.token_pool[0, req.device_len : req.cached_len + 4].tolist() == [
        3,
        0,
        0,
    ]


def test_scheduler_batches_variable_drafts_into_one_staging_buffer():
    reqs = [_make_req(0, [1, 2, 3, 1, 2]), _make_req(1, [4, 5, 6])]
    batch = Batch(
        reqs=reqs,
        phase="verify",
        draft_ids=[[7, 8], []],
        verify_width=3,
    )
    batch.padded_reqs = batch.reqs
    scheduler: Any = object.__new__(Scheduler)
    scheduler.engine = SimpleNamespace(max_seq_len=16)
    scheduler.token_pool = torch.full((2, 16), 99, dtype=torch.int32)
    scheduler.device = torch.device("cpu")
    scheduler._draft_staging_buffer_index = 0
    scheduler._draft_staging_buffers = [
        _DraftStagingBuffer.create(4, scheduler.device, pin_memory=False)
    ]

    scheduler._stage_drafts(batch)

    assert scheduler.token_pool[0, reqs[0].device_len : reqs[0].device_len + 2].tolist() == [
        7,
        8,
    ]
    assert scheduler.token_pool[1, reqs[1].device_len : reqs[1].device_len + 2].tolist() == [
        0,
        0,
    ]


def test_scheduler_reuses_mapping_buffers_for_verify_rows():
    reqs = [_make_req(0, [1, 2, 3]), _make_req(1, [4, 5, 6])]
    batch = Batch(
        reqs=reqs,
        phase="verify",
        draft_ids=[[7], []],
        verify_width=3,
    )
    batch.padded_reqs = batch.reqs
    scheduler: Any = object.__new__(Scheduler)
    scheduler.device = torch.device("cpu")
    scheduler._mapping_staging_buffer_index = 0
    scheduler._mapping_staging_buffers = [
        _MappingStagingBuffer.create(6, 2, scheduler.device, pin_memory=False)
    ]

    input_mapping, write_mapping = scheduler._prepare_mappings(batch)

    assert input_mapping[0].tolist() == [0, 0, 0, 1, 1, 1]
    assert input_mapping[1].tolist() == [2, 3, 4, 2, 3, 4]
    assert batch.positions.tolist() == [2, 3, 4, 2, 3, 4]
    assert len(write_mapping[0]) == len(write_mapping[1]) == 0


def test_scheduler_maps_padding_rows_to_dummy_kv_page():
    req = _make_req(0, [1, 2, 3, 1, 2])
    batch = Batch(
        reqs=[req],
        phase="verify",
        draft_ids=[[3]],
        verify_width=4,
    )
    batch.padded_reqs = batch.reqs
    page_table = torch.full((2, 16), -1, dtype=torch.int32)
    page_table[1].fill_(123)
    scheduler: Any = object.__new__(Scheduler)
    scheduler.engine = SimpleNamespace(
        page_table=page_table,
        dummy_req=SimpleNamespace(table_idx=1),
    )

    scheduler._stage_verify_padding(batch)

    assert batch.allocated_device_len(0) == 6
    assert batch.forward_device_len(0) == 8
    assert page_table[0, 6:8].tolist() == [123, 123]


def test_scheduler_reconciles_padded_verification_rows_and_frees_padding_kv(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    speculator = NgramSpeculator(ngram_size=2, num_draft_tokens=3)
    reqs = [
        _make_req(0, [1, 2, 3, 1, 2]),
        _make_req(1, [4, 5, 6, 4, 5]),
    ]
    batch = Batch(
        reqs=reqs,
        phase="verify",
        draft_ids=[[3], [6, 4]],
        verify_width=4,
    )
    batch.padded_reqs = batch.reqs
    scheduler: Any = object.__new__(Scheduler)
    scheduler.speculator = speculator
    scheduler.cache_manager = Mock()
    scheduler.cache_manager.lazy_free_region.return_value = nullcontext()
    scheduler.decode_manager = Mock()
    scheduler.finished_reqs = set()
    scheduler.eos_token_id = -1
    scheduler.token_pool = torch.zeros((2, 16), dtype=torch.int32)
    scheduler.token_pool[0, 5] = 3
    scheduler.token_pool[1, 5:7] = torch.tensor([6, 4], dtype=torch.int32)
    scheduler.device = torch.device("cpu")
    scheduler._pending_token_staging_buffer_index = 0
    scheduler._pending_token_staging_buffers = [
        _PendingTokenStagingBuffer.create(2, scheduler.device, pin_memory=False)
    ]
    scheduler.send_result = Mock()

    target_tokens = torch.tensor(
        [
            3,
            9,
            99,
            99,  # request 0 padding
            8,
            99,
            99,
            99,  # request 1 target tail and padding
        ],
        dtype=torch.int32,
    )
    scheduler._process_verify_data(batch, target_tokens, target_tokens.clone())

    assert reqs[0].input_ids[-2:].tolist() == [3, 9]
    assert reqs[1].input_ids[-1:].tolist() == [8]
    assert reqs[0].cached_len == 6 and reqs[0].device_len == 7
    assert reqs[1].cached_len == 5 and reqs[1].device_len == 6
    assert scheduler.token_pool[0, 5:7].tolist() == [3, 9]
    assert scheduler.token_pool[1, 5:7].tolist() == [8, 4]
    scheduler.cache_manager.free_req_suffixes.assert_called_once_with(
        reqs,
        [6, 5],
        [6, 7],
    )


def _make_decode_result(batch: Batch, token: int):
    copy_done = Mock()
    return (
        SimpleNamespace(batch=batch),
        (None, torch.tensor([token], dtype=torch.int32), copy_done),
    )


def _make_result_scheduler() -> Any:
    scheduler: Any = object.__new__(Scheduler)
    scheduler.cache_manager = Mock()
    scheduler.cache_manager.lazy_free_region.return_value = nullcontext()
    scheduler.decode_manager = Mock()
    scheduler.finished_reqs = set()
    scheduler.discarded_inflight_reqs = set()
    scheduler.eos_token_id = 99
    scheduler.send_result = Mock()
    scheduler._free_req_resources = Mock()
    return scheduler


def test_overlap_max_length_waits_for_the_inflight_token() -> None:
    req = _make_req(0, [1, 2, 3])
    req.sampling_params.ignore_eos = True
    req.max_device_len = req.device_len + 2
    batch = Batch(reqs=[req], phase="decode")
    req.complete_one()
    req.complete_one()
    scheduler = _make_result_scheduler()

    scheduler._process_last_data(_make_decode_result(batch, 7), {req})
    first_reply = scheduler.send_result.call_args.args[0]
    assert len(req.input_ids) == 4
    assert not first_reply[0].finished
    scheduler._process_last_data(_make_decode_result(batch, 8))
    second_reply = scheduler.send_result.call_args.args[0]

    assert req.input_ids.tolist() == [1, 2, 3, 7, 8]
    assert second_reply[0].finished
    scheduler._free_req_resources.assert_called_once_with(req)


def test_overlap_discards_a_forward_launched_past_eos() -> None:
    req = _make_req(0, [1, 2, 3])
    req.max_device_len = req.device_len + 2
    batch = Batch(reqs=[req], phase="decode")
    req.complete_one()
    req.complete_one()
    scheduler = _make_result_scheduler()
    scheduler.eos_token_id = 7

    scheduler._process_last_data(_make_decode_result(batch, 7), {req})
    first_reply = scheduler.send_result.call_args.args[0]
    assert first_reply[0].finished
    assert req in scheduler.discarded_inflight_reqs
    scheduler._process_last_data(_make_decode_result(batch, 8))

    assert req.input_ids.tolist() == [1, 2, 3, 7]
    assert req not in scheduler.discarded_inflight_reqs
    scheduler._free_req_resources.assert_called_once_with(req)


def test_speculator_reserves_one_output_token_for_the_bonus_token():
    speculator = NgramSpeculator(ngram_size=2, num_draft_tokens=3)
    req = _make_req(0, [1, 2, 3, 1, 2])
    req.max_device_len = req.device_len + 2

    batch = speculator.schedule([req])

    assert batch is not None and batch.is_verify
    assert batch.draft_ids is not None and batch.draft_ids[0] == [3]


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
    stats.record_lookup(matched=True, match_size=3)
    stats.record_lookup(matched=True, match_size=1)
    stats.record_lookup(matched=False)
    stats.record_lookup(matched=False)
    stats.record_verify(drafted_tokens=3, accepted_drafts=3)
    stats.record_verify(drafted_tokens=3, accepted_drafts=1)
    stats.record_verify(drafted_tokens=2, accepted_drafts=0)

    assert stats.lookup_attempts == 4
    assert stats.lookup_matches == 2
    assert stats.lookup_misses == 2
    assert stats.lookup_match_rate == 1 / 2
    assert stats.lookup_matches_by_size == {3: 1, 1: 1}
    assert stats.position_attempts == [3, 2, 1]
    assert stats.position_accepts == [2, 1, 1]
