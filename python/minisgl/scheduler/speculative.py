from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, List, Protocol, Sequence

import torch
from minisgl.core import Batch, Req

if TYPE_CHECKING:
    from minisgl.engine.sample import BatchSamplingArgs, Sampler

    from .config import SchedulerConfig


def find_ngram_draft(
    input_ids: torch.Tensor,
    ngram_size: int,
    max_draft_tokens: int,
) -> torch.Tensor:
    """Return a continuation from the longest matching suffix, up to ngram_size."""
    draft, _ = _find_ngram_draft(input_ids, ngram_size, max_draft_tokens)
    return draft


def _find_ngram_draft(
    input_ids: torch.Tensor,
    ngram_size: int,
    max_draft_tokens: int,
) -> tuple[torch.Tensor, int]:
    assert input_ids.is_cpu and input_ids.ndim == 1
    if ngram_size <= 0 or max_draft_tokens <= 0 or len(input_ids) <= 1:
        return torch.empty(0, dtype=input_ids.dtype), 0

    max_match_size = min(ngram_size, len(input_ids) - 1)
    candidate_ends = torch.nonzero(input_ids[:-1] == input_ids[-1]).flatten().flip(0)
    if not len(candidate_ends):
        return torch.empty(0, dtype=input_ids.dtype), 0

    offsets = torch.arange(max_match_size)
    candidate_indices = candidate_ends[:, None] - offsets[None, :]
    valid_indices = candidate_indices >= 0
    candidate_tokens = input_ids[candidate_indices.clamp_min(0)]
    suffix_tokens = input_ids[-1 - offsets]
    matching_tokens = valid_indices & (candidate_tokens == suffix_tokens)
    match_sizes = matching_tokens.to(torch.int32).cumprod(dim=1).sum(dim=1)

    # Candidates are newest-to-oldest, so argmax keeps the most recent
    # occurrence when multiple candidates have the same longest match.
    best_index = int(match_sizes.argmax().item())
    best_match_size = int(match_sizes[best_index].item())
    best_end = int(candidate_ends[best_index].item())
    continuation = input_ids[best_end + 1 : best_end + 1 + max_draft_tokens]
    return continuation.clone(), best_match_size


@dataclass
class _NgramHistoryIndex:
    _TOKEN_BITS = 32
    _TOKEN_BASE = 1 << _TOKEN_BITS

    max_ngram_size: int
    tokens: List[int]
    occurrences: List[dict[int, int]]

    @classmethod
    def build(cls, input_ids: torch.Tensor, max_ngram_size: int) -> _NgramHistoryIndex:
        index = cls(
            max_ngram_size=max_ngram_size,
            tokens=input_ids.tolist(),
            occurrences=[{} for _ in range(max_ngram_size + 1)],
        )
        index._index_ends(0, len(index.tokens) - 1)
        return index

    def sync(self, input_ids: torch.Tensor) -> None:
        old_len = len(self.tokens)
        assert len(input_ids) >= old_len
        if len(input_ids) == old_len:
            return
        self.append_tokens(input_ids[old_len:].tolist())

    def append_tokens(self, token_ids: List[int]) -> None:
        if not token_ids:
            return
        old_len = len(self.tokens)
        self.tokens.extend(token_ids)
        # The old final token and every newly appended non-final token now
        # have a known continuation and can become lookup candidates.
        self._index_ends(old_len - 1, len(self.tokens) - 1)

    def _index_ends(self, start: int, stop: int) -> None:
        for end in range(max(start, 0), max(stop, 0)):
            key = 0
            multiplier = 1
            for size in range(1, min(self.max_ngram_size, end + 1) + 1):
                token = self.tokens[end - size + 1]
                assert 0 <= token < self._TOKEN_BASE
                key += token * multiplier
                self.occurrences[size][key] = end
                multiplier *= self._TOKEN_BASE

    def find(self, max_draft_tokens: int) -> tuple[List[int], int]:
        max_match_size = min(self.max_ngram_size, len(self.tokens) - 1)
        suffix_keys = [0] * (max_match_size + 1)
        key = 0
        multiplier = 1
        for size in range(1, max_match_size + 1):
            token = self.tokens[-size]
            assert 0 <= token < self._TOKEN_BASE
            key += token * multiplier
            suffix_keys[size] = key
            multiplier *= self._TOKEN_BASE
        for size in range(max_match_size, 0, -1):
            end = self.occurrences[size].get(suffix_keys[size])
            if end is None:
                continue
            continuation = self.tokens[end + 1 : end + 1 + max_draft_tokens]
            if continuation:
                return continuation, size
        return [], 0


@dataclass(frozen=True)
class VerificationResult:
    token_ids: torch.Tensor
    accepted_drafts: int
    emitted_token_ids: List[int]


def accept_deterministic_draft(
    draft_ids: Sequence[int] | torch.Tensor, target_tokens: torch.Tensor
) -> VerificationResult:
    """Accept a matching deterministic draft prefix and one target token."""
    assert target_tokens.is_cpu and target_tokens.ndim == 1
    if isinstance(draft_ids, torch.Tensor):
        assert draft_ids.is_cpu and draft_ids.ndim == 1
        draft_values: Sequence[int] = draft_ids.tolist()
    else:
        draft_values = draft_ids
    assert len(target_tokens) == len(draft_ids) + 1

    target_values = target_tokens.tolist()
    accepted = 0
    for draft_token, target_token in zip(draft_values, target_values[:-1], strict=True):
        if draft_token != target_token:
            break
        accepted += 1
    # Every accepted draft equals its target token, so the emitted sequence is
    # already a contiguous target prefix. Keep a view instead of concatenating.
    return VerificationResult(
        token_ids=target_tokens[: accepted + 1],
        accepted_drafts=accepted,
        emitted_token_ids=target_values[: accepted + 1],
    )


class _RankLogger(Protocol):
    def info_rank0(self, msg: str, *args: object) -> None: ...


class SpeculativeStrategy(Protocol):
    """Scheduler-facing contract for the current linear verification flow."""

    @property
    def cuda_graph_verify_width(self) -> int: ...

    def schedule(self, reqs: Iterable[Req]) -> Batch | None: ...

    def prepare_sampling(self, batch: Batch, sampler: Sampler) -> BatchSamplingArgs: ...

    def select_verification_tokens(
        self,
        batch: Batch,
        logits: torch.Tensor,
        sampler: Sampler,
        args: BatchSamplingArgs,
    ) -> torch.Tensor: ...

    def verify(
        self, batch: Batch, index: int, target_tokens: torch.Tensor
    ) -> VerificationResult: ...

    def verify_batch(
        self, batch: Batch, target_tokens: torch.Tensor
    ) -> List[VerificationResult]: ...

    def record_verification(self, batch: Batch, index: int, accepted_drafts: int) -> None: ...

    def update_history(self, req: Req, token_ids: List[int]) -> None: ...

    def release(self, req: Req) -> None: ...

    def log_stats(self, logger: _RankLogger) -> None: ...


@dataclass
class SpeculativeStats:
    lookup_attempts: int = 0
    lookup_matches: int = 0
    lookup_matches_by_size: dict[int, int] = field(default_factory=dict)
    verify_steps: int = 0
    drafted_tokens: int = 0
    accepted_drafts: int = 0
    position_attempts: List[int] = field(default_factory=list)
    position_accepts: List[int] = field(default_factory=list)
    verify_batches: int = 0
    decode_batches: int = 0
    verify_rows: int = 0
    decode_rows: int = 0
    folded_decode_rows: int = 0

    def record_lookup(self, matched: bool, match_size: int = 0) -> None:
        assert match_size >= 0
        assert matched or match_size == 0
        self.lookup_attempts += 1
        self.lookup_matches += int(matched)
        if match_size:
            self.lookup_matches_by_size[match_size] = (
                self.lookup_matches_by_size.get(match_size, 0) + 1
            )

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
class NgramSpeculator(SpeculativeStrategy):
    ngram_size: int
    num_draft_tokens: int
    mixed_batch: bool = False
    stats: SpeculativeStats = field(default_factory=SpeculativeStats)
    _prefer_verify: bool = True
    _history_indices: dict[Req, _NgramHistoryIndex] = field(default_factory=dict)

    @property
    def cuda_graph_verify_width(self) -> int:
        # One pending token followed by the configured draft window.
        return self.num_draft_tokens + 1

    def _draft(self, req: Req) -> List[int]:
        if req.remain_len <= 1:
            return []
        max_draft_tokens = min(self.num_draft_tokens, req.remain_len - 1)
        index = self._history_indices.get(req)
        if index is None:
            index = _NgramHistoryIndex.build(req.input_ids, self.ngram_size)
            self._history_indices[req] = index
        else:
            index.sync(req.input_ids)
        draft, match_size = index.find(max_draft_tokens)
        self.stats.record_lookup(matched=bool(len(draft)), match_size=match_size)
        return draft

    def release(self, req: Req) -> None:
        self._history_indices.pop(req, None)

    def update_history(self, req: Req, token_ids: List[int]) -> None:
        index = self._history_indices.get(req)
        if index is not None:
            index.append_tokens(token_ids)

    def schedule(self, reqs: Iterable[Req]) -> Batch | None:
        ordered = sorted(reqs, key=lambda req: req.uid)
        if not ordered:
            return None

        verify_reqs: List[Req] = []
        draft_ids: List[List[int]] = []
        normal_reqs: List[Req] = []
        ordered_drafts: List[List[int]] = []
        for req in ordered:
            draft = self._draft(req)
            ordered_drafts.append(draft)
            if len(draft):
                assert req.extend_len == 1
                verify_reqs.append(req)
                draft_ids.append(draft)
            else:
                normal_reqs.append(req)

        if self.mixed_batch:
            self.stats.verify_batches += 1
            self.stats.verify_rows += len(ordered)
            self.stats.folded_decode_rows += len(normal_reqs)
            return Batch(reqs=ordered, phase="verify", draft_ids=ordered_drafts)

        if verify_reqs and normal_reqs:
            use_verify = self._prefer_verify
            self._prefer_verify = not self._prefer_verify
        else:
            use_verify = bool(verify_reqs)

        if use_verify:
            self.stats.verify_batches += 1
            self.stats.verify_rows += len(verify_reqs)
            return Batch(reqs=verify_reqs, phase="verify", draft_ids=draft_ids)
        self.stats.decode_batches += 1
        self.stats.decode_rows += len(normal_reqs)
        return Batch(reqs=normal_reqs, phase="decode")

    def prepare_sampling(self, batch: Batch, sampler: Sampler) -> BatchSamplingArgs:
        assert batch.is_verify
        if all(req.sampling_params.is_greedy for req in batch.reqs):
            return sampler.prepare_params([batch.reqs[0].sampling_params])
        params = [
            req.sampling_params
            for i, req in enumerate(batch.reqs)
            for _ in range(batch.forward_extend_len(i))
        ]
        return sampler.prepare_params(params)

    def select_verification_tokens(
        self,
        batch: Batch,
        logits: torch.Tensor,
        sampler: Sampler,
        args: BatchSamplingArgs,
    ) -> torch.Tensor:
        """Sample target tokens for deterministic-proposal rejection sampling.

        An n-gram draft has proposal probability one for its candidate token.
        Drawing from the target distribution therefore implements exact
        rejection sampling: equality accepts the draft, while a mismatch is
        already a sample from the correct residual distribution.
        """
        assert batch.is_verify
        return sampler.sample(logits, args)

    def verify(self, batch: Batch, index: int, target_tokens: torch.Tensor) -> VerificationResult:
        assert batch.is_verify and batch.draft_ids is not None
        verify_len = batch.verification_len(index)
        return accept_deterministic_draft(batch.draft_ids[index], target_tokens[:verify_len])

    def verify_batch(self, batch: Batch, target_tokens: torch.Tensor) -> List[VerificationResult]:
        assert batch.is_verify and batch.draft_ids is not None
        target_values = target_tokens.tolist()
        results: List[VerificationResult] = []
        offset = 0
        for i, draft_ids in enumerate(batch.draft_ids):
            accepted = 0
            for j, draft_token in enumerate(draft_ids):
                if draft_token != target_values[offset + j]:
                    break
                accepted += 1
            results.append(
                VerificationResult(
                    token_ids=target_tokens[offset : offset + accepted + 1],
                    accepted_drafts=accepted,
                    emitted_token_ids=target_values[offset : offset + accepted + 1],
                )
            )
            offset += batch.forward_extend_len(i)
        assert offset == len(target_tokens)
        return results

    def record_verification(self, batch: Batch, index: int, accepted_drafts: int) -> None:
        assert batch.is_verify and batch.draft_ids is not None
        drafted_tokens = len(batch.draft_ids[index])
        if drafted_tokens:
            self.stats.record_verify(drafted_tokens, accepted_drafts)

    def log_stats(self, logger: _RankLogger) -> None:
        stats = self.stats
        logger.info_rank0(
            "N-gram lookup: attempts=%d, matches=%d, misses=%d, match_rate=%.2f%%",
            stats.lookup_attempts,
            stats.lookup_matches,
            stats.lookup_misses,
            100 * stats.lookup_match_rate,
        )
        matches_by_size = ", ".join(
            f"n{size}={count}"
            for size, count in sorted(stats.lookup_matches_by_size.items(), reverse=True)
        )
        logger.info_rank0(
            "N-gram lookup matches by suffix length: %s",
            matches_by_size or "none",
        )
        logger.info_rank0(
            "N-gram verification: verify_steps=%d, drafted_tokens=%d, "
            "accepted_drafts=%d, mean_accepted_drafts=%.2f",
            stats.verify_steps,
            stats.drafted_tokens,
            stats.accepted_drafts,
            stats.mean_accepted_drafts,
        )
        position_rates = ", ".join(
            f"p{i}={accepted}/{attempts} ({100 * accepted / attempts:.2f}%)"
            for i, (attempts, accepted) in enumerate(
                zip(stats.position_attempts, stats.position_accepts, strict=True)
            )
            if attempts > 0
        )
        logger.info_rank0(
            "N-gram conditional acceptance by draft position: %s",
            position_rates or "none",
        )
        logger.info_rank0(
            "N-gram scheduling: verify_batches=%d, decode_batches=%d, "
            "verify_rows=%d, decode_rows=%d, folded_decode_rows=%d",
            stats.verify_batches,
            stats.decode_batches,
            stats.verify_rows,
            stats.decode_rows,
            stats.folded_decode_rows,
        )


def _parse_ngram_config(raw_config: str) -> tuple[int, int]:
    try:
        parsed = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        raise ValueError("--spec-decoding-config must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--spec-decoding-config must be a JSON object.")

    expected_keys = {"ngram_size", "num_draft_tokens"}
    actual_keys = set(parsed)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unknown = sorted(actual_keys - expected_keys)
        details: List[str] = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown keys: {', '.join(unknown)}")
        raise ValueError(f"Invalid ngram speculative config ({'; '.join(details)}).")

    ngram_size = parsed["ngram_size"]
    num_draft_tokens = parsed["num_draft_tokens"]
    if type(ngram_size) is not int or ngram_size <= 0:
        raise ValueError("ngram_size must be a positive integer.")
    if type(num_draft_tokens) is not int or num_draft_tokens <= 0:
        raise ValueError("num_draft_tokens must be a positive integer.")
    return ngram_size, num_draft_tokens


def _create_speculator(config: SchedulerConfig) -> SpeculativeStrategy | None:
    algorithm = config.spec_decoding
    raw_config = config.spec_decoding_config
    if algorithm is None:
        if raw_config is not None:
            raise ValueError("--spec-decoding-config requires --spec-decoding.")
        return None
    if algorithm != "ngram":
        raise ValueError(f"Unsupported speculative decoding algorithm: {algorithm!r}.")
    if raw_config is None:
        raise ValueError("--spec-decoding ngram requires --spec-decoding-config.")

    ngram_size, num_draft_tokens = _parse_ngram_config(raw_config)
    if config.tp_info.size != 1:
        raise ValueError("N-gram speculation currently requires tensor parallel size 1.")
    if config.page_size != 1:
        raise ValueError("N-gram speculation currently requires --page-size 1.")
    if config.attention_backend not in ("fa", "fi"):
        raise ValueError("N-gram speculation currently requires --attention-backend fa or fi.")
    return NgramSpeculator(
        ngram_size,
        num_draft_tokens,
        mixed_batch=True,
    )
