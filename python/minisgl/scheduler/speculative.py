from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, List, Protocol

import torch
from minisgl.core import Batch, Req
from minisgl.env import ENV

if TYPE_CHECKING:
    from minisgl.engine.sample import BatchSamplingArgs, Sampler

    from .config import SchedulerConfig


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
class VerificationResult:
    token_ids: torch.Tensor
    accepted_drafts: int


def accept_deterministic_draft(
    draft_ids: torch.Tensor, target_tokens: torch.Tensor
) -> VerificationResult:
    """Accept a matching deterministic draft prefix and one target token."""
    assert draft_ids.is_cpu and target_tokens.is_cpu
    assert draft_ids.ndim == target_tokens.ndim == 1
    assert len(target_tokens) == len(draft_ids) + 1

    mismatch = torch.nonzero(target_tokens[:-1] != draft_ids)
    accepted = int(mismatch[0].item()) if len(mismatch) else len(draft_ids)
    token_ids = torch.cat(
        [draft_ids[:accepted], target_tokens[accepted : accepted + 1]]
    )
    return VerificationResult(token_ids=token_ids, accepted_drafts=accepted)


class _RankLogger(Protocol):
    def info_rank0(self, msg: str, *args: object) -> None: ...


class SpeculativeStrategy(Protocol):
    """Scheduler-facing contract for the current linear verification flow."""

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

    def record_verification(
        self, batch: Batch, index: int, accepted_drafts: int
    ) -> None: ...

    def log_stats(self, logger: _RankLogger) -> None: ...


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
class NgramSpeculator(SpeculativeStrategy):
    ngram_size: int
    num_draft_tokens: int
    stats: SpeculativeStats = field(default_factory=SpeculativeStats)
    _prefer_verify: bool = True

    def _draft(self, req: Req) -> torch.Tensor:
        if req.remain_len <= 1:
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

    def prepare_sampling(self, batch: Batch, sampler: Sampler) -> BatchSamplingArgs:
        assert batch.is_verify
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

    def verify(
        self, batch: Batch, index: int, target_tokens: torch.Tensor
    ) -> VerificationResult:
        assert batch.is_verify and batch.draft_ids is not None
        return accept_deterministic_draft(batch.draft_ids[index], target_tokens)

    def record_verification(
        self, batch: Batch, index: int, accepted_drafts: int
    ) -> None:
        assert batch.is_verify and batch.draft_ids is not None
        self.stats.record_verify(len(batch.draft_ids[index]), accepted_drafts)

    def log_stats(self, logger: _RankLogger) -> None:
        stats = self.stats
        logger.info_rank0(
            "N-gram lookup: attempts=%d, matches=%d, misses=%d, match_rate=%.2f%%",
            stats.lookup_attempts,
            stats.lookup_matches,
            stats.lookup_misses,
            100 * stats.lookup_match_rate,
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
        raise ValueError(
            "N-gram speculation currently requires --attention-backend fa or fi."
        )
    if not ENV.DISABLE_OVERLAP_SCHEDULING:
        raise ValueError(
            "N-gram speculation currently requires MINISGL_DISABLE_OVERLAP_SCHEDULING=1."
        )
    return NgramSpeculator(ngram_size, num_draft_tokens)
