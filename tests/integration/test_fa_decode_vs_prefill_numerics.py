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


def _top2(logits: Any) -> tuple[int, int, float]:
    import torch

    prediction = int(torch.argmax(logits).item())
    runner_up_logits = logits.float().clone()
    selected_logit = float(runner_up_logits[prediction].item())
    runner_up_logits[prediction] = -torch.inf
    runner_up = int(torch.argmax(runner_up_logits).item())
    margin = selected_logit - float(runner_up_logits[runner_up].item())
    return prediction, runner_up, margin


class _PrefillProbeTrace:
    def __init__(self, probes: list[dict[str, Any]]):
        self.probes_by_uid = {index: probe for index, probe in enumerate(probes)}
        self.rows: dict[int, dict[str, Any]] = {}

    def record(self, batch: Any, logits: Any) -> None:
        if not batch.is_prefill:
            return
        assert len(logits) >= batch.size
        for index, req in enumerate(batch.reqs):
            probe = self.probes_by_uid[req.uid]
            row = logits[index]
            prediction, runner_up, margin = _top2(row)
            self.rows[req.uid] = {
                "prefill_token": prediction,
                "prefill_runner_up": runner_up,
                "prefill_top2_margin": margin,
                "prefill_selected_logprob": _token_logprob(row, prediction),
                "prefill_baseline_token_logprob": _token_logprob(
                    row, probe["baseline_token"]
                ),
                "prefill_speculative_token_logprob": _token_logprob(
                    row, probe["speculative_token"]
                ),
            }


def _run_prefill_worker(
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
        "prefill",
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
            f"prefill worker failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return json.loads(result_path.read_text())


def test_fa3_prefill_reproduces_cnn_decode_divergences(tmp_path: Path) -> None:
    """Probe FA3 prefill-vs-decode numerics on CNN cases that spec decoding exposes."""

    if os.environ.get("MINISGL_RUN_FA3_DIVERGENCE_REPRO") != "1":
        pytest.skip("set MINISGL_RUN_FA3_DIVERGENCE_REPRO=1 to run the H100 repro")

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
    prefill_path = tmp_path / "prefill.json"
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
        probes.append(
            {
                "case_index": case_index,
                "case_id": baseline["case_ids"][case_index],
                "article": cases[case_index]["article"],
                "position": common_len,
                "prefix_token_ids": base_tokens[:common_len],
                "baseline_token": base_tokens[common_len],
                "speculative_token": spec_tokens[common_len],
                "decode_selected_logprob": baseline["logprobs"][case_index][common_len],
                "decode_runner_up": baseline["runner_up_ids"][case_index][common_len],
                "decode_top2_margin": baseline["top2_margins"][case_index][common_len],
                "spec_selected_logprob": speculative["logprobs"][case_index][common_len],
                "spec_runner_up": speculative["runner_up_ids"][case_index][common_len],
                "spec_top2_margin": speculative["top2_margins"][case_index][common_len],
            }
        )
        if len(probes) >= divergence_cases:
            break

    assert probes, "the CNN run did not expose any speculative divergences"
    probes_path.write_text(json.dumps(probes))
    prefill = _run_prefill_worker(
        probes_path,
        prefill_path,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        batch_size=batch_size,
    )

    rows = prefill["rows"]
    baseline_deltas: list[float] = []
    prefill_token_matches_spec = 0
    prefill_token_matches_decode = 0
    examples: list[dict[str, Any]] = []
    for probe, row in zip(probes, rows, strict=True):
        baseline_delta = abs(
            row["prefill_baseline_token_logprob"] - probe["decode_selected_logprob"]
        )
        baseline_deltas.append(baseline_delta)
        prefill_token_matches_spec += int(row["prefill_token"] == probe["speculative_token"])
        prefill_token_matches_decode += int(row["prefill_token"] == probe["baseline_token"])
        examples.append(
            {
                "case_index": probe["case_index"],
                "case_id": probe["case_id"],
                "position": probe["position"],
                "decode_token": probe["baseline_token"],
                "prefill_token": row["prefill_token"],
                "speculative_token": probe["speculative_token"],
                "decode_logprob": probe["decode_selected_logprob"],
                "prefill_decode_token_logprob": row["prefill_baseline_token_logprob"],
                "prefill_spec_token_logprob": row["prefill_speculative_token_logprob"],
                "abs_decode_token_logprob_delta": baseline_delta,
                "decode_runner_up": probe["decode_runner_up"],
                "prefill_runner_up": row["prefill_runner_up"],
                "decode_top2_margin": probe["decode_top2_margin"],
                "prefill_top2_margin": row["prefill_top2_margin"],
            }
        )

    summary = {
        "cnn_cases": num_cases,
        "divergence_cases": len(probes),
        "ngram_size": speculative["ngram_size"],
        "num_draft_tokens": speculative["num_draft_tokens"],
        "prefill_token_matches_decode": prefill_token_matches_decode,
        "prefill_token_matches_speculative": prefill_token_matches_spec,
        "max_abs_decode_token_logprob_delta": max(baseline_deltas),
        "mean_abs_decode_token_logprob_delta": sum(baseline_deltas)
        / len(baseline_deltas),
        "p99_abs_decode_token_logprob_delta": _percentile(baseline_deltas, 0.99),
        "logprob_atol": logprob_atol,
        "first_examples": examples[:10],
    }
    print(json.dumps(summary, indent=2))

    assert max(baseline_deltas) <= logprob_atol, summary
    assert prefill_token_matches_decode == len(probes), summary


def _prefill_worker(args: argparse.Namespace) -> None:
    import torch

    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    probes = json.loads(Path(args.probes).read_text())
    model = os.environ.get("MINISGL_CNN_MODEL", "Qwen/Qwen3-0.6B")
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
    trace = _PrefillProbeTrace(probes)
    original_forward = llm.engine.model.forward

    def traced_forward() -> Any:
        logits = original_forward()
        trace.record(llm.engine.ctx.batch, logits)
        return logits

    llm.engine.model.forward = traced_forward
    rows: list[dict[str, Any]] = []
    try:
        for start in range(0, len(probes), args.batch_size):
            batch_probes = probes[start : start + args.batch_size]
            prompt_ids = _tokenize_articles(
                llm.tokenizer,
                [probe["article"] for probe in batch_probes],
                args.max_input_tokens,
            )
            prompt_ids = [
                ids + probe["prefix_token_ids"]
                for ids, probe in zip(prompt_ids, batch_probes, strict=True)
            ]
            trace.rows.clear()
            results = llm.generate(
                prompt_ids,
                SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=1),
            )
            for uid, result in enumerate(results):
                output_ids = result["token_ids"]
                assert isinstance(output_ids, list) and len(output_ids) == 1
                row = trace.rows[uid]
                assert row["prefill_token"] == output_ids[0]
                rows.append(row)
    finally:
        llm.shutdown()

    Path(args.result).write_text(json.dumps({"rows": rows, "gpu": torch.cuda.get_device_name()}))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("prefill",))
    parser.add_argument("--probes", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = _parse_args()
    if parsed.worker == "prefill":
        if parsed.probes is None or parsed.result is None:
            raise SystemExit("--probes and --result are required")
        _prefill_worker(parsed)
