from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

import torch
from minisgl.core import Batch, Req


def find_ngram_draft(
    input_ids: torch.Tensor,
    ngram_size: int,
    max_draft_tokens: int,
) -> torch.Tensor:
    """Return the continuation of the most recent earlier suffix match."""
    assert input_ids.is_cpu and input_ids.ndim == 1
    if ngram_size <= 0 or max_draft_tokens <= 0 or len(input_ids) <= ngram_size:
        return torch.empty(0, dtype=input_ids.dtype)

    tokens = input_ids.tolist()
    suffix = tokens[-ngram_size:]
    for start in range(len(tokens) - ngram_size - 1, -1, -1):
        if tokens[start : start + ngram_size] != suffix:
            continue
        continuation = tokens[start + ngram_size : start + ngram_size + max_draft_tokens]
        if continuation:
            return torch.tensor(continuation, dtype=input_ids.dtype)
    return torch.empty(0, dtype=input_ids.dtype)


@dataclass(frozen=True)
class GreedyAcceptance:
    token_ids: torch.Tensor
    accepted_drafts: int


def greedy_accept(draft_ids: torch.Tensor, predictions: torch.Tensor) -> GreedyAcceptance:
    """Accept a matching draft prefix and one target-model token."""
    assert draft_ids.is_cpu and predictions.is_cpu
    assert draft_ids.ndim == predictions.ndim == 1
    assert len(predictions) == len(draft_ids) + 1

    mismatch = torch.nonzero(predictions[:-1] != draft_ids)
    accepted = int(mismatch[0].item()) if len(mismatch) else len(draft_ids)
    token_ids = torch.cat([draft_ids[:accepted], predictions[accepted : accepted + 1]])
    return GreedyAcceptance(token_ids=token_ids, accepted_drafts=accepted)


@dataclass
class SpeculativeStats:
    lookup_attempts: int = 0
    lookup_matches: int = 0
    verify_steps: int = 0
    drafted_tokens: int = 0
    accepted_drafts: int = 0
    position_attempts: List[int] = field(default_factory=list)
    position_accepts: List[int] = field(default_factory=list)

    def record_lookup(self, matched: bool) -> None:
        self.lookup_attempts += 1
        self.lookup_matches += int(matched)

    def record_verify(self, drafted_tokens: int, accepted_drafts: int) -> None:
        assert 0 <= accepted_drafts <= drafted_tokens
        self.verify_steps += 1
        self.drafted_tokens += drafted_tokens
        self.accepted_drafts += accepted_drafts
        while len(self.position_attempts) < drafted_tokens:
            self.position_attempts.append(0)
            self.position_accepts.append(0)
        for position in range(drafted_tokens):
            if accepted_drafts >= position:
                self.position_attempts[position] += 1
            if accepted_drafts > position:
                self.position_accepts[position] += 1

    @property
    def lookup_misses(self) -> int:
        return self.lookup_attempts - self.lookup_matches

    @property
    def lookup_match_rate(self) -> float:
        return self.lookup_matches / self.lookup_attempts if self.lookup_attempts else 0.0

    @property
    def mean_accepted_drafts(self) -> float:
        return self.accepted_drafts / self.verify_steps if self.verify_steps else 0.0


@dataclass
class NgramSpeculator:
    ngram_size: int
    num_draft_tokens: int
    stats: SpeculativeStats = field(default_factory=SpeculativeStats)
    _prefer_verify: bool = True

    def _draft(self, req: Req) -> torch.Tensor:
        if not req.sampling_params.is_greedy or req.remain_len <= 1:
            return torch.empty(0, dtype=req.input_ids.dtype)
        max_draft_tokens = min(self.num_draft_tokens, req.remain_len - 1)
        draft = find_ngram_draft(req.input_ids, self.ngram_size, max_draft_tokens)
        self.stats.record_lookup(matched=bool(len(draft)))
        return draft

    def schedule(self, reqs: Iterable[Req]) -> Batch | None:
        ordered = sorted(reqs, key=lambda req: req.uid)
        if not ordered:
            return None

        verify_reqs: List[Req] = []
        draft_ids: List[torch.Tensor] = []
        normal_reqs: List[Req] = []
        for req in ordered:
            draft = self._draft(req)
            if len(draft):
                assert req.extend_len == 1
                verify_reqs.append(req)
                draft_ids.append(draft)
            else:
                normal_reqs.append(req)

        if verify_reqs and normal_reqs:
            use_verify = self._prefer_verify
            self._prefer_verify = not self._prefer_verify
        else:
            use_verify = bool(verify_reqs)

        if use_verify:
            return Batch(reqs=verify_reqs, phase="verify", draft_ids=draft_ids)
        return Batch(reqs=normal_reqs, phase="decode")
