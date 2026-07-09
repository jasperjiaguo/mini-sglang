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
    DEFAULT_BATCH_SIZE,
    DEFAULT_LOGPROB_ATOL,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_NUM_CASES,
    DEFAULT_NGRAM_SIZE,
    DEFAULT_NUM_DRAFT_TOKENS,
    _load_cnn_articles,
    _percentile,
    _run_worker,
    _tokenize_articles,
)


DEFAULT_DIVERGENCE_CASES = 50


def _token_logprob(logits: Any, token_id: int) -> float:
    import torch

    float_logits = logits.float()
    return float((float_logits[token_id] - torch.logsumexp(float_logits, dim=-1)).item())


def _row_stats(logits: Any, probe: dict[str, Any], prefix: str) -> dict[str, Any]:
    import torch

    prediction = int(torch.argmax(logits).item())
    runner_up_logits = logits.float().clone()
    selected_logit = float(runner_up_logits[prediction].item())
    runner_up_logits[prediction] = -torch.inf
    runner_up = int(torch.argmax(runner_up_logits).item())
    margin = selected_logit - float(runner_up_logits[runner_up].item())
    return {
        f"{prefix}_token": prediction,
        f"{prefix}_runner_up": runner_up,
        f"{prefix}_top2_margin": margin,
        f"{prefix}_selected_logprob": _token_logprob(logits, prediction),
        f"{prefix}_baseline_token_logprob": _token_logprob(logits, probe["baseline_token"]),
        f"{prefix}_speculative_token_logprob": _token_logprob(
            logits, probe["speculative_token"]
        ),
    }


def _run_extend_worker(
    cases_path: Path,
    probes_path: Path,
    result_path: Path,
    *,
    max_input_tokens: int,
    max_output_tokens: int,
    batch_size: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "extend",
        "--cases",
        str(cases_path),
        "--probes",
        str(probes_path),
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
    completed = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise AssertionError(
            f"extend worker failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return json.loads(result_path.read_text())


def test_fa3_extend_matches_decode_on_cnn_divergences(tmp_path: Path) -> None:
    """Probe FA3 cached-extend-vs-decode numerics where n-gram spec diverges."""

    if os.environ.get("MINISGL_RUN_FA3_EXTEND_REPRO") != "1":
        pytest.skip("set MINISGL_RUN_FA3_EXTEND_REPRO=1 to run the H100 repro")

    num_cases = int(os.environ.get("MINISGL_CNN_CASES", DEFAULT_NUM_CASES))
    divergence_cases = int(
        os.environ.get("MINISGL_DIVERGENCE_CASES", DEFAULT_DIVERGENCE_CASES)
    )
    max_input_tokens = int(
        os.environ.get("MINISGL_CNN_MAX_INPUT_TOKENS", DEFAULT_MAX_INPUT_TOKENS)
    )
    max_output_tokens = int(
        os.environ.get("MINISGL_CNN_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)
    )
    batch_size = int(os.environ.get("MINISGL_CNN_BATCH_SIZE", DEFAULT_BATCH_SIZE))
    logprob_atol = float(os.environ.get("MINISGL_LOGPROB_ATOL", DEFAULT_LOGPROB_ATOL))

    cases = _load_cnn_articles(num_cases)
    cases_path = tmp_path / "cnn_cases.json"
    baseline_path = tmp_path / "baseline.json"
    speculative_path = tmp_path / "speculative.json"
    probes_path = tmp_path / "probes.json"
    extend_path = tmp_path / "extend.json"
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

    probes: list[dict[str, Any]] = []
    for case_index, (base_tokens, spec_tokens) in enumerate(
        zip(baseline["token_ids"], speculative["token_ids"], strict=True)
    ):
        common_len = 0
        for base_token, spec_token in zip(base_tokens, spec_tokens):
            if base_token != spec_token:
                break
            common_len += 1
        if base_tokens == spec_tokens or common_len >= len(base_tokens):
            continue
        event = speculative["token_events"][case_index][common_len]
        # Only verify-emitted divergences have a spec-shaped cached extend to replay.
        if event["phase"] != "verify" or event["event_start_position"] == 0:
            continue
        assert event["event_start_position"] <= common_len
        probes.append(
            {
                "case_index": case_index,
                "case_id": baseline["case_ids"][case_index],
                "article": cases[case_index]["article"],
                "position": common_len,
                "prefix_token_ids": base_tokens[:common_len],
                "event_start_position": event["event_start_position"],
                "event_prefix_token_ids": base_tokens[: event["event_start_position"]],
                "row_offset": event["row_offset"],
                "draft_ids": event["draft_ids"],
                "spec_verify_predictions": event["verify_predictions"],
                "spec_accepted_drafts": event["accepted_drafts"],
                "spec_accepted_len": event["accepted_len"],
                "baseline_token": base_tokens[common_len],
                "speculative_token": spec_tokens[common_len],
                "baseline_trace_logprob": baseline["logprobs"][case_index][common_len],
                "baseline_trace_runner_up": baseline["runner_up_ids"][case_index][common_len],
                "baseline_trace_top2_margin": baseline["top2_margins"][case_index][common_len],
                "spec_trace_logprob": speculative["logprobs"][case_index][common_len],
                "spec_trace_runner_up": speculative["runner_up_ids"][case_index][common_len],
                "spec_trace_top2_margin": speculative["top2_margins"][case_index][common_len],
            }
        )
        if len(probes) >= divergence_cases:
            break

    assert probes, "the CNN run did not expose any comparable speculative divergences"
    probes_path.write_text(json.dumps(probes))
    extend = _run_extend_worker(
        cases_path,
        probes_path,
        extend_path,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        batch_size=batch_size,
    )

    rows = extend["rows"]
    deltas: list[float] = []
    verify_token_matches_spec = 0
    verify_token_matches_decode = 0
    decode_token_matches_baseline_trace = 0
    examples: list[dict[str, Any]] = []
    for probe, row in zip(probes, rows, strict=True):
        delta = abs(
            row["verify_baseline_token_logprob"] - row["decode_baseline_token_logprob"]
        )
        deltas.append(delta)
        verify_token_matches_spec += int(row["verify_token"] == probe["speculative_token"])
        verify_token_matches_decode += int(row["verify_token"] == row["decode_token"])
        decode_token_matches_baseline_trace += int(
            row["decode_token"] == probe["baseline_token"]
        )
        examples.append(
            {
                "case_index": probe["case_index"],
                "case_id": probe["case_id"],
                "position": probe["position"],
                "event_start_position": probe["event_start_position"],
                "row_offset": probe["row_offset"],
                "decode_token": row["decode_token"],
                "baseline_trace_token": probe["baseline_token"],
                "verify_token": row["verify_token"],
                "verify_emitted_token": row["verify_emitted_token"],
                "speculative_token": probe["speculative_token"],
                "draft_ids": row["draft_ids"],
                "accepted_drafts": row["accepted_drafts"],
                "decode_baseline_token_logprob": row["decode_baseline_token_logprob"],
                "verify_baseline_token_logprob": row["verify_baseline_token_logprob"],
                "verify_spec_token_logprob": row["verify_speculative_token_logprob"],
                "abs_baseline_token_logprob_delta": delta,
                "decode_runner_up": row["decode_runner_up"],
                "verify_runner_up": row["verify_runner_up"],
                "decode_top2_margin": row["decode_top2_margin"],
                "verify_top2_margin": row["verify_top2_margin"],
            }
        )

    summary = {
        "cnn_cases": num_cases,
        "divergence_cases": len(probes),
        "ngram_size": extend["ngram_size"],
        "num_draft_tokens": extend["num_draft_tokens"],
        "decode_token_matches_baseline_trace": decode_token_matches_baseline_trace,
        "verify_token_matches_decode": verify_token_matches_decode,
        "verify_token_matches_speculative": verify_token_matches_spec,
        "max_abs_baseline_token_logprob_delta": max(deltas),
        "mean_abs_baseline_token_logprob_delta": sum(deltas) / len(deltas),
        "p99_abs_baseline_token_logprob_delta": _percentile(deltas, 0.99),
        "logprob_atol": logprob_atol,
        "first_examples": examples[:10],
    }
    print(json.dumps(summary, indent=2))

    assert max(deltas) <= logprob_atol, summary
    assert verify_token_matches_decode == len(probes), summary


class _ShadowExtendTrace:
    def __init__(self, llm: Any):
        self.llm = llm
        self.probes_by_uid: dict[int, dict[str, Any]] = {}
        self.decode_rows: dict[int, dict[str, Any]] = {}
        self.verify_rows: dict[int, dict[str, Any]] = {}

    def set_probes(self, probes_by_uid: dict[int, dict[str, Any]]) -> None:
        self.probes_by_uid = probes_by_uid
        self.decode_rows.clear()
        self.verify_rows.clear()

    def _is_at_decode_position(self, req: Any) -> bool:
        probe = self.probes_by_uid.get(req.uid)
        if probe is None:
            return False
        status = self.llm.status_map.get(req.uid)
        return status is not None and len(status.output_ids) == probe["position"]

    def _is_at_verify_position(self, req: Any) -> bool:
        probe = self.probes_by_uid.get(req.uid)
        if probe is None:
            return False
        status = self.llm.status_map.get(req.uid)
        return (
            status is not None
            and len(status.output_ids) == probe["event_start_position"]
        )

    def record(self, batch: Any, logits: Any) -> None:
        if batch.is_verify:
            self._record_verify(batch, logits)
            return

        for index, req in enumerate(batch.reqs):
            if not self._is_at_decode_position(req):
                continue
            probe = self.probes_by_uid[req.uid]
            self.decode_rows[req.uid] = _row_stats(logits[index], probe, "decode")

    def _record_verify(self, batch: Any, logits: Any) -> None:
        import torch

        from minisgl.scheduler.speculative import greedy_accept

        assert batch.draft_ids is not None
        offset = 0
        for index, (req, draft_ids) in enumerate(zip(batch.reqs, batch.draft_ids, strict=True)):
            verify_len = batch.forward_extend_len(index)
            req_logits = logits[offset : offset + verify_len]
            offset += verify_len
            if not self._is_at_verify_position(req):
                continue

            probe = self.probes_by_uid[req.uid]
            row_offset = probe["row_offset"]
            assert row_offset < verify_len
            predictions = torch.argmax(req_logits, dim=-1).to(torch.int32).cpu()
            acceptance = greedy_accept(draft_ids, predictions)
            row = _row_stats(req_logits[row_offset], probe, "verify")
            verify_emitted_token = (
                int(acceptance.token_ids[row_offset].item())
                if row_offset < len(acceptance.token_ids)
                else None
            )
            row.update(
                {
                    "draft_ids": draft_ids.tolist(),
                    "verify_predictions": predictions.tolist(),
                    "accepted_drafts": acceptance.accepted_drafts,
                    "accepted_len": len(acceptance.token_ids),
                    "verify_emitted_token": verify_emitted_token,
                }
            )
            self.verify_rows[req.uid] = row
        assert offset == len(logits)


def _run_shadow_verify_if_needed(
    llm: Any,
    trace: _ShadowExtendTrace,
    *,
    ngram_size: int,
    num_draft_tokens: int,
) -> None:
    import torch

    from minisgl.core import Batch

    target_reqs = [
        req
        for req in sorted(llm.decode_manager.running_reqs, key=lambda r: r.uid)
        if req.uid in trace.probes_by_uid
        and req.uid not in trace.verify_rows
        and trace._is_at_verify_position(req)
    ]
    if not target_reqs:
        return

    verify_reqs = []
    draft_ids = []
    for req in target_reqs:
        if not req.sampling_params.is_greedy or req.remain_len <= 1:
            continue
        probe = trace.probes_by_uid[req.uid]
        draft = torch.tensor(probe["draft_ids"], dtype=req.input_ids.dtype)
        if len(draft):
            verify_reqs.append(req)
            draft_ids.append(draft)
    if not verify_reqs:
        return

    batch = Batch(reqs=verify_reqs, phase="verify", draft_ids=draft_ids)
    forward_input = llm._prepare_batch(batch)
    _, _, copy_done = llm._forward(forward_input)
    copy_done.synchronize()
    with llm.cache_manager.lazy_free_region():
        for index, req in enumerate(batch.reqs):
            llm.cache_manager.free_req_suffix(
                req,
                start=req.cached_len,
                end=batch.forward_device_len(index),
            )


def _cleanup_active_requests(llm: Any) -> None:
    llm.pending_requests = []
    llm.prefill_manager.pending_list.clear()
    for req in list(llm.decode_manager.running_reqs):
        llm.decode_manager.remove_req(req)
        llm._free_req_resources(req)
    llm.status_map = {}
    llm.counter = 0
    llm.finished_reqs = set()


def _extend_worker(args: argparse.Namespace) -> None:
    import torch

    from minisgl.core import SamplingParams
    from minisgl.llm import LLM
    from minisgl.llm.llm import RequestAllFinished

    cases = json.loads(Path(args.cases).read_text())
    probes = json.loads(Path(args.probes).read_text())
    probes_by_case_index = {probe["case_index"]: probe for probe in probes}
    model = os.environ.get("MINISGL_CNN_MODEL", "Qwen/Qwen3-0.6B")
    ngram_size = int(os.environ.get("MINISGL_NGRAM_SIZE", DEFAULT_NGRAM_SIZE))
    num_draft_tokens = int(
        os.environ.get("MINISGL_NUM_DRAFT_TOKENS", DEFAULT_NUM_DRAFT_TOKENS)
    )
    llm = LLM(
        model,
        attention_backend="fa",
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
    trace = _ShadowExtendTrace(llm)
    original_forward = llm.engine.model.forward

    def traced_forward() -> Any:
        logits = original_forward()
        trace.record(llm.engine.ctx.batch, logits)
        return logits

    llm.engine.model.forward = traced_forward
    rows_by_case_index: dict[int, dict[str, Any]] = {}
    try:
        with llm.engine_stream_ctx:
            llm.engine.stream.wait_stream(llm.stream)
            for start in range(0, len(cases), args.batch_size):
                batch_cases = cases[start : start + args.batch_size]
                probes_by_uid = {
                    uid: probes_by_case_index[start + uid]
                    for uid in range(len(batch_cases))
                    if start + uid in probes_by_case_index
                }
                if not probes_by_uid:
                    continue
                prompt_ids = _tokenize_articles(
                    llm.tokenizer,
                    [case["article"] for case in batch_cases],
                    args.max_input_tokens,
                )
                trace.set_probes(probes_by_uid)
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

                while len(trace.verify_rows) < len(probes_by_uid):
                    _run_shadow_verify_if_needed(
                        llm,
                        trace,
                        ngram_size=ngram_size,
                        num_draft_tokens=num_draft_tokens,
                    )
                    try:
                        llm.normal_loop()
                    except RequestAllFinished as exc:
                        missing_verify = sorted(set(probes_by_uid) - set(trace.verify_rows))
                        raise AssertionError(
                            f"generation finished before all probes were recorded: "
                            f"{missing_verify=}"
                        ) from exc

                for uid, probe in probes_by_uid.items():
                    status = llm.status_map[uid]
                    event_prefix = status.output_ids[: probe["event_start_position"]]
                    assert event_prefix == probe["event_prefix_token_ids"]
                    row = dict(trace.verify_rows[uid])
                    row.update(
                        {
                            "decode_token": probe["baseline_token"],
                            "decode_runner_up": probe["baseline_trace_runner_up"],
                            "decode_top2_margin": probe["baseline_trace_top2_margin"],
                            "decode_selected_logprob": probe["baseline_trace_logprob"],
                            "decode_baseline_token_logprob": probe[
                                "baseline_trace_logprob"
                            ],
                            "decode_speculative_token_logprob": None,
                        }
                    )
                    rows_by_case_index[probe["case_index"]] = row
                _cleanup_active_requests(llm)
    finally:
        llm.shutdown()

    Path(args.result).write_text(
        json.dumps(
            {
                "rows": [rows_by_case_index[probe["case_index"]] for probe in probes],
                "ngram_size": ngram_size,
                "num_draft_tokens": num_draft_tokens,
                "gpu": torch.cuda.get_device_name(),
            }
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("extend",))
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--probes", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = _parse_args()
    if parsed.worker == "extend":
        if parsed.cases is None or parsed.probes is None or parsed.result is None:
            raise SystemExit("--cases, --probes, and --result are required")
        _extend_worker(parsed)
