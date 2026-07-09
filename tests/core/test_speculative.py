from __future__ import annotations

import torch

from minisgl.core import Batch, Req, SamplingParams
from minisgl.scheduler.speculative import (
    NgramSpeculator,
    SpeculativeStats,
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
