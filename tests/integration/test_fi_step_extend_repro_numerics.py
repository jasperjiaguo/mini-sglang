from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.integration.test_ngram_speculative_numerics import (
    DEFAULT_LOGPROB_ATOL,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_NGRAM_SIZE,
    DEFAULT_NUM_CASES,
    DEFAULT_NUM_DRAFT_TOKENS,
    _load_cnn_articles,
    _percentile,
    _run_worker,
    _tokenize_articles,
)


DEFAULT_STEP_EXTEND_ATTENTION_BACKEND = "fi"
DEFAULT_STEP_EXTEND_BATCH_SIZE = 1


def _selected_attention_backend() -> str:
    backend = os.environ.get(
        "MINISGL_STEP_EXTEND_BACKEND",
        os.environ.get(
            "MINISGL_ATTENTION_BACKEND", DEFAULT_STEP_EXTEND_ATTENTION_BACKEND
        ),
    ).strip().lower()
    aliases = {
        "fa3": "fa",
        "flashattention3": "fa",
        "flash-attention-3": "fa",
        "flash_attention_3": "fa",
        "flashinfer": "fi",
    }
    backend = aliases.get(backend, backend)
    if backend not in {"fi", "fa"}:
        raise ValueError(
            f"unsupported step-extend attention backend {backend!r}; expected 'fi' "
            "or 'fa'"
        )
    return backend


def _token_logprob(logits: Any, token_id: int) -> float:
    import torch

    float_logits = logits.float()
    return float((float_logits[token_id] - torch.logsumexp(float_logits, dim=-1)).item())


def _row_stats(logits: Any, baseline_token: int, speculative_token: int) -> dict[str, Any]:
    import torch

    float_logits = logits.float()
    prediction = int(torch.argmax(float_logits).item())
    runner_up_logits = float_logits.clone()
    selected_logit = float(runner_up_logits[prediction].item())
    runner_up_logits[prediction] = -torch.inf
    runner_up = int(torch.argmax(runner_up_logits).item())
    return {
        "extend_token": prediction,
        "extend_runner_up": runner_up,
        "extend_top2_margin": selected_logit - float(runner_up_logits[runner_up].item()),
        "extend_selected_logprob": _token_logprob(logits, prediction),
        "extend_baseline_token_logprob": _token_logprob(logits, baseline_token),
        "extend_speculative_token_logprob": _token_logprob(logits, speculative_token),
    }


def _first_mismatch(base_tokens: list[int], spec_tokens: list[int]) -> int | None:
    for position, (base_token, spec_token) in enumerate(zip(base_tokens, spec_tokens)):
        if base_token != spec_token:
            return position
    if base_tokens != spec_tokens:
        return min(len(base_tokens), len(spec_tokens))
    return None


def _unique_spec_steps(
    token_events: list[dict[str, Any]],
    *,
    prompt_len: int,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    seen_verify_starts: set[int] = set()
    for position, event in enumerate(token_events):
        phase = event["phase"]
        if phase == "verify":
            start = event["event_start_position"]
            if start in seen_verify_starts:
                continue
            seen_verify_starts.add(start)
            draft_len = len(event["draft_ids"])
            steps.append(
                {
                    "position": start,
                    "phase": "verify",
                    "prefill_len": 1 + draft_len,
                    "draft_len": draft_len,
                    "accepted_len": event["accepted_len"],
                    "accepted_drafts": event["accepted_drafts"],
                }
            )
        else:
            steps.append(
                {
                    "position": position,
                    "phase": phase,
                    "prefill_len": prompt_len if phase == "prefill" else 1,
                    "draft_len": 0,
                    "accepted_len": 1,
                    "accepted_drafts": 0,
                }
            )
    return steps


def _verify_plans(token_events: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    plans: dict[int, dict[str, Any]] = {}
    for event in token_events:
        if event["phase"] != "verify":
            continue
        start = event["event_start_position"]
        plans.setdefault(
            start,
            {
                "event_start_position": start,
                "draft_ids": event["draft_ids"],
                "accepted_len": event["accepted_len"],
                "accepted_drafts": event["accepted_drafts"],
            },
        )
    return plans


def _run_step_extend_worker(
    cases_path: Path,
    payload_path: Path,
    result_path: Path,
    *,
    max_input_tokens: int,
    max_output_tokens: int,
    batch_size: int,
    attention_backend: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "step-extend",
        "--cases",
        str(cases_path),
        "--payload",
        str(payload_path),
        "--result",
        str(result_path),
        "--max-input-tokens",
        str(max_input_tokens),
        "--max-output-tokens",
        str(max_output_tokens),
        "--batch-size",
        str(batch_size),
    ]
    env = os.environ.copy()
    env["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = "1"
    env["MINISGL_ATTENTION_BACKEND"] = attention_backend
    completed = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise AssertionError(
            f"step-extend worker failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return json.loads(result_path.read_text())


def test_fi_step_extend_reproduces_mismatch_rows(tmp_path: Path) -> None:
    """Replay n-gram mismatch rows as no-spec forced step-extend probes."""

    if (
        os.environ.get("MINISGL_RUN_STEP_EXTEND_REPRO") != "1"
        and os.environ.get("MINISGL_RUN_FI_STEP_EXTEND_REPRO") != "1"
    ):
        pytest.skip(
            "set MINISGL_RUN_STEP_EXTEND_REPRO=1 to run the H100 repro "
            "(MINISGL_RUN_FI_STEP_EXTEND_REPRO=1 is also accepted)"
        )

    attention_backend = _selected_attention_backend()
    old_backend = os.environ.get("MINISGL_ATTENTION_BACKEND")
    os.environ["MINISGL_ATTENTION_BACKEND"] = attention_backend
    try:
        num_cases = int(os.environ.get("MINISGL_CNN_CASES", DEFAULT_NUM_CASES))
        max_input_tokens = int(
            os.environ.get("MINISGL_CNN_MAX_INPUT_TOKENS", DEFAULT_MAX_INPUT_TOKENS)
        )
        max_output_tokens = int(
            os.environ.get("MINISGL_CNN_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)
        )
        batch_size = int(
            os.environ.get("MINISGL_CNN_BATCH_SIZE", DEFAULT_STEP_EXTEND_BATCH_SIZE)
        )
        logprob_atol = float(os.environ.get("MINISGL_LOGPROB_ATOL", DEFAULT_LOGPROB_ATOL))

        cases = _load_cnn_articles(num_cases)
        cases_path = tmp_path / "cnn_cases.json"
        baseline_path = tmp_path / "baseline.json"
        speculative_path = tmp_path / "speculative.json"
        payload_path = tmp_path / "payload.json"
        step_extend_path = tmp_path / "step_extend.json"
        cases_path.write_text(json.dumps(cases))

        baseline = _run_worker(
            "baseline",
            cases_path,
            baseline_path,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            batch_size=batch_size,
        )
        speculative = _run_worker(
            "speculative",
            cases_path,
            speculative_path,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            batch_size=batch_size,
        )
    finally:
        if old_backend is None:
            os.environ.pop("MINISGL_ATTENTION_BACKEND", None)
        else:
            os.environ["MINISGL_ATTENTION_BACKEND"] = old_backend

    assert baseline["case_ids"] == speculative["case_ids"]

    # Tokenize once in the parent for human-readable step summaries.
    from transformers import AutoTokenizer

    model = os.environ.get("MINISGL_CNN_MODEL", "Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(model)
    prompt_ids = _tokenize_articles(
        tokenizer,
        [case["article"] for case in cases],
        max_input_tokens,
    )

    mismatch_cases: list[dict[str, Any]] = []
    case_reports: list[dict[str, Any]] = []
    all_plans: dict[str, dict[str, Any]] = {}
    baseline_tokens_by_case: dict[str, list[int]] = {}

    for case_index, (base_tokens, spec_tokens) in enumerate(
        zip(baseline["token_ids"], speculative["token_ids"], strict=True)
    ):
        all_plans[str(case_index)] = _verify_plans(
            speculative["token_events"][case_index]
        )
        baseline_tokens_by_case[str(case_index)] = base_tokens
        mismatch_position = _first_mismatch(base_tokens, spec_tokens)
        if mismatch_position is None:
            continue

        event = speculative["token_events"][case_index][mismatch_position]
        phase = event["phase"]
        draft_ids = event.get("draft_ids", [])

        mismatch_case = {
            "case_index": case_index,
            "case_id": baseline["case_ids"][case_index],
            "position": mismatch_position,
            "mismatch_phase": phase,
            "event_start_position": event["event_start_position"],
            "row_offset": event["row_offset"],
            "baseline_token": base_tokens[mismatch_position],
            "speculative_token": spec_tokens[mismatch_position],
            "baseline_logprob": baseline["logprobs"][case_index][mismatch_position],
            "baseline_runner_up": baseline["runner_up_ids"][case_index][mismatch_position],
            "baseline_top2_margin": baseline["top2_margins"][case_index][mismatch_position],
            "draft_ids": draft_ids,
            "prefill_len": 1 + len(draft_ids) if phase == "verify" else 1,
            "spec_accepted_len": event.get("accepted_len", 1),
            "spec_accepted_drafts": event.get("accepted_drafts", 0),
        }
        mismatch_cases.append(mismatch_case)
        case_reports.append(
            {
                **mismatch_case,
                "prompt_len": len(prompt_ids[case_index]),
                "spec_final_sequence": spec_tokens,
                "spec_steps": _unique_spec_steps(
                    speculative["token_events"][case_index],
                    prompt_len=len(prompt_ids[case_index]),
                ),
            }
        )

    payload_path.write_text(
        json.dumps(
            {
                "mismatch_cases": mismatch_cases,
                "verify_plans": all_plans,
                "baseline_tokens": baseline_tokens_by_case,
            }
        )
    )

    step_extend = _run_step_extend_worker(
        cases_path,
        payload_path,
        step_extend_path,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        batch_size=batch_size,
        attention_backend=attention_backend,
    )
    rows = step_extend["rows"]

    deltas = [
        abs(row["extend_baseline_token_logprob"] - row["baseline_logprob"])
        for row in rows
    ]
    summary = {
        "backend": attention_backend,
        "cnn_cases": num_cases,
        "mismatch_cases": len(mismatch_cases),
        "step_extend_rows": len(rows),
        "step_extend_token_matches_autoregressive": sum(
            row["extend_token"] == row["baseline_token"] for row in rows
        ),
        "step_extend_token_matches_speculative": sum(
            row["extend_token"] == row["speculative_token"] for row in rows
        ),
        "numerical_diff_cases_over_atol": sum(delta > logprob_atol for delta in deltas),
        "max_abs_ar_vs_step_baseline_logprob_delta": max(deltas),
        "mean_abs_ar_vs_step_baseline_logprob_delta": sum(deltas) / len(deltas),
        "p99_abs_ar_vs_step_baseline_logprob_delta": _percentile(deltas, 0.99),
        "logprob_atol": logprob_atol,
        "rows": rows,
        "spec_case_reports": case_reports,
    }
    print(json.dumps(summary, indent=2))

    assert rows
    assert len(rows) == len(mismatch_cases)


def _cleanup_active_requests(llm: Any) -> None:
    llm.pending_requests = []
    llm.prefill_manager.pending_list.clear()
    for req in list(llm.decode_manager.running_reqs):
        llm.decode_manager.remove_req(req)
        llm._free_req_resources(req)
    llm.status_map = {}
    llm.counter = 0
    llm.finished_reqs = set()


class _StepExtendReplayTrace:
    def __init__(
        self,
        llm: Any,
        *,
        mismatch_by_case: dict[int, dict[str, Any]],
        batch_start: int,
    ) -> None:
        self.llm = llm
        self.mismatch_by_case = mismatch_by_case
        self.batch_start = batch_start
        self.rows: dict[int, dict[str, Any]] = {}

    def set_batch_start(self, batch_start: int) -> None:
        self.batch_start = batch_start

    def record(self, batch: Any, logits: Any) -> None:
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if batch.is_verify:
            self._record_verify(batch, logits)
            return
        self._record_decode_or_prefill(batch, logits)

    def _record_decode_or_prefill(self, batch: Any, logits: Any) -> None:
        from minisgl.scheduler.prefill import ChunkedReq

        for index, req in enumerate(batch.reqs):
            if isinstance(req, ChunkedReq):
                continue
            case_index = self.batch_start + req.uid
            mismatch = self.mismatch_by_case.get(case_index)
            if mismatch is None or case_index in self.rows:
                continue
            if mismatch["mismatch_phase"] not in ("decode", "prefill"):
                continue
            status = self.llm.status_map.get(req.uid)
            if status is None or len(status.output_ids) != mismatch["position"]:
                continue
            row = _row_stats(
                logits[index],
                mismatch["baseline_token"],
                mismatch["speculative_token"],
            )
            row.update(self._row_context(mismatch, batch.forward_extend_len(index)))
            self.rows[case_index] = row

    def _record_verify(self, batch: Any, logits: Any) -> None:
        assert batch.draft_ids is not None
        offset = 0
        for index, req in enumerate(batch.reqs):
            verify_len = batch.forward_extend_len(index)
            req_logits = logits[offset : offset + verify_len]
            offset += verify_len
            case_index = self.batch_start + req.uid
            mismatch = self.mismatch_by_case.get(case_index)
            if mismatch is None or case_index in self.rows:
                continue
            if mismatch["mismatch_phase"] != "verify":
                continue
            status = self.llm.status_map.get(req.uid)
            if status is None:
                continue
            position = len(status.output_ids)
            if position != mismatch["event_start_position"]:
                continue
            row_offset = mismatch["row_offset"]
            row = _row_stats(
                req_logits[row_offset],
                mismatch["baseline_token"],
                mismatch["speculative_token"],
            )
            row.update(self._row_context(mismatch, verify_len))
            self.rows[case_index] = row
        assert offset == len(logits)

    @staticmethod
    def _row_context(mismatch: dict[str, Any], prefill_len: int) -> dict[str, Any]:
        return {
            "case_index": mismatch["case_index"],
            "case_id": mismatch["case_id"],
            "position": mismatch["position"],
            "mismatch_phase": mismatch["mismatch_phase"],
            "event_start_position": mismatch["event_start_position"],
            "row_offset": mismatch["row_offset"],
            "prefill_len": prefill_len,
            "draft_ids": mismatch["draft_ids"],
            "spec_accepted_len": mismatch["spec_accepted_len"],
            "spec_accepted_drafts": mismatch["spec_accepted_drafts"],
            "baseline_token": mismatch["baseline_token"],
            "speculative_token": mismatch["speculative_token"],
            "baseline_logprob": mismatch["baseline_logprob"],
            "baseline_runner_up": mismatch["baseline_runner_up"],
            "baseline_top2_margin": mismatch["baseline_top2_margin"],
        }


def _should_force_plan(
    *,
    case_index: int,
    position: int,
    plan: dict[str, Any] | None,
    mismatch_positions: dict[int, int],
) -> bool:
    if plan is None:
        return False
    mismatch_position = mismatch_positions.get(case_index)
    return mismatch_position is not None and position <= mismatch_position


def _run_forced_step_extend_if_needed(
    llm: Any,
    *,
    batch_start: int,
    verify_plans: dict[str, dict[str, Any]],
    baseline_tokens: dict[str, list[int]],
    mismatch_by_case: dict[int, dict[str, Any]],
    mismatch_positions: dict[int, int],
    trace: _StepExtendReplayTrace,
) -> bool:
    import torch

    from minisgl.core import Batch

    selected: list[tuple[Any, int, dict[str, Any]]] = []
    for req in sorted(llm.decode_manager.running_reqs, key=lambda r: r.uid):
        case_index = batch_start + req.uid
        if case_index in trace.rows:
            continue
        status = llm.status_map.get(req.uid)
        if status is None:
            continue
        position = len(status.output_ids)
        plan = verify_plans.get(str(case_index), {}).get(str(position))
        if _should_force_plan(
            case_index=case_index,
            position=position,
            plan=plan,
            mismatch_positions=mismatch_positions,
        ):
            selected.append((req, case_index, plan))

    if not selected:
        return False

    draft_ids = [
        torch.tensor(plan["draft_ids"], dtype=req.input_ids.dtype)
        for req, _, plan in selected
    ]
    batch = Batch(reqs=[req for req, _, _ in selected], phase="verify", draft_ids=draft_ids)
    forward_input = llm._prepare_batch(batch)
    _, _, copy_done = llm._forward(forward_input)
    copy_done.synchronize()

    with llm.cache_manager.lazy_free_region():
        for local_index, (req, case_index, plan) in enumerate(selected):
            status = llm.status_map[req.uid]
            position = len(status.output_ids)
            mismatch = mismatch_by_case[case_index]
            mismatch_position = mismatch["position"]

            if position <= mismatch_position < position + plan["accepted_len"]:
                row_offset = mismatch_position - position
                advance_len = row_offset + 1
            else:
                advance_len = plan["accepted_len"]

            baseline = baseline_tokens[str(case_index)]
            token_ids = torch.tensor(
                baseline[position : position + advance_len],
                dtype=req.input_ids.dtype,
            )
            assert len(token_ids) == advance_len
            new_cached_len = req.cached_len + advance_len
            llm.cache_manager.free_req_suffix(
                req,
                start=new_cached_len,
                end=batch.forward_device_len(local_index),
            )

            output_start = req.device_len
            req.cached_len = new_cached_len
            req.device_len += advance_len
            req.append_host(token_ids)
            assert req.cached_len + 1 == req.device_len == len(req.input_ids)
            llm.token_pool[req.table_idx, output_start : req.device_len].copy_(
                token_ids.pin_memory(), non_blocking=True
            )
            status.output_ids.extend(token_ids.tolist())

            if not req.can_decode:
                llm.decode_manager.remove_req(req)
                llm._free_req_resources(req)
    return True


def _step_extend_worker(args: argparse.Namespace) -> None:
    import torch

    from minisgl.core import SamplingParams
    from minisgl.llm import LLM
    from minisgl.llm.llm import RequestAllFinished

    cases = json.loads(Path(args.cases).read_text())
    payload = json.loads(Path(args.payload).read_text())
    mismatch_cases = payload["mismatch_cases"]
    mismatch_by_case = {case["case_index"]: case for case in mismatch_cases}
    mismatch_positions = {
        case["case_index"]: case["position"] for case in mismatch_cases
    }
    verify_plans = payload["verify_plans"]
    baseline_tokens = payload["baseline_tokens"]
    model = os.environ.get("MINISGL_CNN_MODEL", "Qwen/Qwen3-0.6B")
    attention_backend = _selected_attention_backend()

    llm = LLM(
        model,
        attention_backend=attention_backend,
        cache_type="naive",
        cuda_graph_max_bs=0,
        max_extend_tokens=args.batch_size * args.max_input_tokens + 128,
        max_running_req=args.batch_size,
        max_seq_len_override=args.max_input_tokens + args.max_output_tokens + 8,
        num_page_override=(args.max_input_tokens + args.max_output_tokens + 8)
        * args.batch_size
        * 2,
        page_size=1,
    )
    trace = _StepExtendReplayTrace(
        llm,
        mismatch_by_case=mismatch_by_case,
        batch_start=0,
    )
    original_forward = llm.engine.model.forward

    def traced_forward() -> Any:
        logits = original_forward()
        trace.record(llm.engine.ctx.batch, logits)
        return logits

    llm.engine.model.forward = traced_forward
    try:
        with llm.engine_stream_ctx:
            llm.engine.stream.wait_stream(llm.stream)
            for start in range(0, len(cases), args.batch_size):
                trace.set_batch_start(start)
                batch_cases = cases[start : start + args.batch_size]
                wanted = {
                    index
                    for index in range(start, start + len(batch_cases))
                    if index in mismatch_by_case
                }
                if not wanted:
                    continue
                prompt_ids = _tokenize_articles(
                    llm.tokenizer,
                    [case["article"] for case in batch_cases],
                    args.max_input_tokens,
                )
                llm.pending_requests = [
                    (
                        prompt,
                        SamplingParams(
                            temperature=0.0,
                            ignore_eos=True,
                            max_tokens=args.max_output_tokens,
                        ),
                    )
                    for prompt in prompt_ids
                ]
                llm.status_map = {}
                llm.counter = 0

                while not wanted.issubset(trace.rows):
                    if _run_forced_step_extend_if_needed(
                        llm,
                        batch_start=start,
                        verify_plans=verify_plans,
                        baseline_tokens=baseline_tokens,
                        mismatch_by_case=mismatch_by_case,
                        mismatch_positions=mismatch_positions,
                        trace=trace,
                    ):
                        continue
                    try:
                        llm.normal_loop()
                    except RequestAllFinished as exc:
                        missing = sorted(wanted - set(trace.rows))
                        raise AssertionError(
                            f"generation finished before all forced rows were recorded: "
                            f"{missing=}"
                        ) from exc
                _cleanup_active_requests(llm)
    finally:
        llm.engine.model.forward = original_forward
        llm.shutdown()

    Path(args.result).write_text(
        json.dumps(
            {
                "rows": [trace.rows[case["case_index"]] for case in mismatch_cases],
                "backend": attention_backend,
                "gpu": torch.cuda.get_device_name(),
            }
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("step-extend",))
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_STEP_EXTEND_BATCH_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = _parse_args()
    if parsed.worker == "step-extend":
        if parsed.cases is None or parsed.payload is None or parsed.result is None:
            raise SystemExit("--cases, --payload, and --result are required")
        _step_extend_worker(parsed)
