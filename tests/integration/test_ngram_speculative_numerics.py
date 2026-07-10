from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


DATASET_REPO = "abisee/cnn_dailymail"
DATASET_CONFIG = "3.0.0"
DATASET_REVISION = "96df5e686bee6baa90b8bee7c28b81fa3fa6223d"
DEFAULT_HF_HOME = "/mnt/mini-sglang-cache/huggingface"
DEFAULT_NUM_CASES = 200
DEFAULT_MAX_INPUT_TOKENS = 768
DEFAULT_MAX_OUTPUT_TOKENS = 32
DEFAULT_FINISH_MAX_OUTPUT_TOKENS = 256
DEFAULT_BATCH_SIZE = 8
DEFAULT_LOGPROB_ATOL = 2e-2
DEFAULT_NGRAM_SIZE = 3
DEFAULT_NUM_DRAFT_TOKENS = 4
DEFAULT_IGNORE_EOS = False


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return _bool_arg(value)


def _load_cnn_articles(num_cases: int) -> list[dict[str, str]]:
    from datasets import load_dataset

    hf_home = os.environ.get("HF_HOME", DEFAULT_HF_HOME)
    dataset = load_dataset(
        DATASET_REPO,
        DATASET_CONFIG,
        split="test",
        revision=DATASET_REVISION,
        cache_dir=f"{hf_home}/datasets",
    )
    indices = random.Random(0).sample(range(len(dataset)), num_cases)
    return [{"id": dataset[index]["id"], "article": dataset[index]["article"]} for index in indices]


def _run_worker(
    mode: str,
    cases_path: Path,
    result_path: Path,
    *,
    max_input_tokens: int,
    max_output_tokens: int,
    batch_size: int,
    ignore_eos: bool,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        mode,
        "--cases",
        str(cases_path),
        "--result",
        str(result_path),
        "--max-input-tokens",
        str(max_input_tokens),
        "--max-output-tokens",
        str(max_output_tokens),
        "--batch-size",
        str(batch_size),
        "--ignore-eos",
        "1" if ignore_eos else "0",
    ]
    env = os.environ.copy()
    env["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = "1"
    completed = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise AssertionError(
            f"{mode} worker failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return json.loads(result_path.read_text())


def _percentile(values: list[float], percentile: float) -> float:
    assert values
    ordered = sorted(values)
    index = math.ceil(percentile * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def _length_stats(lengths: list[int], max_output_tokens: int) -> dict[str, float | int]:
    assert lengths
    return {
        "min": min(lengths),
        "mean": sum(lengths) / len(lengths),
        "max": max(lengths),
        "hit_max_tokens": sum(length == max_output_tokens for length in lengths),
    }


def test_cnn_tokens_and_logprobs_match(tmp_path: Path) -> None:
    """Compare ordinary and n-gram decoding on deterministic CNN articles.

    This test requires Linux, an H100-class CUDA environment, the cached CNN/
    DailyMail dataset, and the Qwen checkpoint. It is intentionally opt-in so
    the normal unit-test suite remains runnable without a GPU.
    """

    if os.environ.get("MINISGL_RUN_CNN_NUMERICS") != "1":
        pytest.skip("set MINISGL_RUN_CNN_NUMERICS=1 to run the H100 numerical test")

    num_cases = int(os.environ.get("MINISGL_CNN_CASES", DEFAULT_NUM_CASES))
    max_input_tokens = int(
        os.environ.get("MINISGL_CNN_MAX_INPUT_TOKENS", DEFAULT_MAX_INPUT_TOKENS)
    )
    max_output_tokens = int(
        os.environ.get(
            "MINISGL_CNN_MAX_OUTPUT_TOKENS", DEFAULT_FINISH_MAX_OUTPUT_TOKENS
        )
    )
    batch_size = int(os.environ.get("MINISGL_CNN_BATCH_SIZE", DEFAULT_BATCH_SIZE))
    logprob_atol = float(os.environ.get("MINISGL_LOGPROB_ATOL", DEFAULT_LOGPROB_ATOL))
    ignore_eos = _env_bool("MINISGL_CNN_IGNORE_EOS", DEFAULT_IGNORE_EOS)

    cases = _load_cnn_articles(num_cases)
    cases_path = tmp_path / "cnn_cases.json"
    baseline_path = tmp_path / "baseline.json"
    speculative_path = tmp_path / "speculative.json"
    cases_path.write_text(json.dumps(cases))

    baseline = _run_worker(
        "baseline",
        cases_path,
        baseline_path,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        batch_size=batch_size,
        ignore_eos=ignore_eos,
    )
    speculative = _run_worker(
        "speculative",
        cases_path,
        speculative_path,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        batch_size=batch_size,
        ignore_eos=ignore_eos,
    )

    assert baseline["case_ids"] == speculative["case_ids"]
    deltas: list[float] = []
    mismatch_details: list[dict[str, Any]] = []
    for case_index, (base_tokens, spec_tokens) in enumerate(
        zip(baseline["token_ids"], speculative["token_ids"], strict=True)
    ):
        common_len = 0
        for base_token, spec_token in zip(base_tokens, spec_tokens):
            if base_token != spec_token:
                break
            common_len += 1
        if base_tokens != spec_tokens:
            mismatch_details.append(
                {
                    "case_index": case_index,
                    "case_id": baseline["case_ids"][case_index],
                    "position": common_len,
                    "baseline_token": (
                        base_tokens[common_len] if common_len < len(base_tokens) else None
                    ),
                    "speculative_token": (
                        spec_tokens[common_len] if common_len < len(spec_tokens) else None
                    ),
                    "baseline_runner_up": (
                        baseline["runner_up_ids"][case_index][common_len]
                        if common_len < len(base_tokens)
                        else None
                    ),
                    "speculative_runner_up": (
                        speculative["runner_up_ids"][case_index][common_len]
                        if common_len < len(spec_tokens)
                        else None
                    ),
                    "baseline_top2_margin": (
                        baseline["top2_margins"][case_index][common_len]
                        if common_len < len(base_tokens)
                        else None
                    ),
                    "speculative_top2_margin": (
                        speculative["top2_margins"][case_index][common_len]
                        if common_len < len(spec_tokens)
                        else None
                    ),
                }
            )

        base_logprobs = baseline["logprobs"][case_index][:common_len]
        spec_logprobs = speculative["logprobs"][case_index][:common_len]
        deltas.extend(
            abs(base - spec)
            for base, spec in zip(base_logprobs, spec_logprobs, strict=True)
        )

    assert deltas
    summary = {
        "cases": num_cases,
        "max_output_tokens": max_output_tokens,
        "ignore_eos": ignore_eos,
        "ngram_size": speculative["ngram_size"],
        "num_draft_tokens": speculative["num_draft_tokens"],
        "baseline_output_length_stats": _length_stats(
            baseline["output_lengths"], max_output_tokens
        ),
        "speculative_output_length_stats": _length_stats(
            speculative["output_lengths"], max_output_tokens
        ),
        "compared_tokens": len(deltas),
        "max_abs_logprob_delta": max(deltas),
        "mean_abs_logprob_delta": sum(deltas) / len(deltas),
        "p99_abs_logprob_delta": _percentile(deltas, 0.99),
        "logprob_atol": logprob_atol,
        "token_mismatch_cases": len(mismatch_details),
        "first_token_mismatches": mismatch_details[:10],
        "speculative_stats": speculative["speculative_stats"],
    }
    print(json.dumps(summary, indent=2))

    assert speculative["speculative_stats"]["verify_steps"] > 0
    assert not mismatch_details, summary
    assert max(deltas) <= logprob_atol, summary


class _ForwardTrace:
    def __init__(self, eos_token_id: int):
        self.eos_token_id = eos_token_id
        self.token_ids: dict[int, list[int]] = {}
        self.logprobs: dict[int, list[float]] = {}
        self.runner_up_ids: dict[int, list[int]] = {}
        self.top2_margins: dict[int, list[float]] = {}
        self.token_events: dict[int, list[dict[str, Any]]] = {}

    def reset(self) -> None:
        self.token_ids.clear()
        self.logprobs.clear()
        self.runner_up_ids.clear()
        self.top2_margins.clear()
        self.token_events.clear()

    def _append(
        self,
        uid: int,
        token_ids: list[int],
        logprobs: list[float],
        runner_up_ids: list[int],
        top2_margins: list[float],
        token_events: list[dict[str, Any]],
    ) -> None:
        assert len(token_ids) == len(logprobs) == len(runner_up_ids) == len(top2_margins)
        assert len(token_ids) == len(token_events)
        self.token_ids.setdefault(uid, []).extend(token_ids)
        self.logprobs.setdefault(uid, []).extend(logprobs)
        self.runner_up_ids.setdefault(uid, []).extend(runner_up_ids)
        self.top2_margins.setdefault(uid, []).extend(top2_margins)
        self.token_events.setdefault(uid, []).extend(token_events)

    def record(self, batch: Any, logits: Any) -> None:
        import torch

        from minisgl.scheduler.prefill import ChunkedReq
        from minisgl.scheduler.speculative import greedy_accept

        float_logits = logits.float()
        predictions = torch.argmax(logits, dim=-1).to(torch.int32)
        prediction_indices = predictions.to(torch.int64).unsqueeze(1)
        selected_logits = float_logits.gather(1, prediction_indices).squeeze(1)
        runner_up_logits = float_logits.clone()
        runner_up_logits.scatter_(1, prediction_indices, -torch.inf)
        runner_up_ids = torch.argmax(runner_up_logits, dim=-1)
        runner_up_values = runner_up_logits.gather(1, runner_up_ids.unsqueeze(1)).squeeze(1)
        selected_logprobs = selected_logits - torch.logsumexp(float_logits, dim=-1)
        top2_margins = selected_logits - runner_up_values
        predictions_cpu = predictions.to("cpu")
        logprobs_cpu = selected_logprobs.to("cpu")
        runner_up_ids_cpu = runner_up_ids.to("cpu")
        top2_margins_cpu = top2_margins.to("cpu")

        if batch.is_verify:
            assert batch.draft_ids is not None
            offset = 0
            for index, (req, draft_ids) in enumerate(
                zip(batch.reqs, batch.draft_ids, strict=True)
            ):
                verify_len = batch.forward_extend_len(index)
                req_predictions = predictions_cpu[offset : offset + verify_len]
                req_logprobs = logprobs_cpu[offset : offset + verify_len]
                req_runner_up_ids = runner_up_ids_cpu[offset : offset + verify_len]
                req_top2_margins = top2_margins_cpu[offset : offset + verify_len]
                offset += verify_len
                acceptance = greedy_accept(draft_ids, req_predictions)
                accepted_len = len(acceptance.token_ids)
                event_start_position = len(self.token_ids.get(req.uid, []))
                token_events = [
                    {
                        "phase": "verify",
                        "event_start_position": event_start_position,
                        "row_offset": row_offset,
                        "draft_ids": draft_ids.tolist(),
                        "verify_predictions": req_predictions.tolist(),
                        "accepted_drafts": acceptance.accepted_drafts,
                        "accepted_len": accepted_len,
                    }
                    for row_offset in range(accepted_len)
                ]
                self._append(
                    req.uid,
                    acceptance.token_ids.tolist(),
                    req_logprobs[:accepted_len].tolist(),
                    req_runner_up_ids[:accepted_len].tolist(),
                    req_top2_margins[:accepted_len].tolist(),
                    token_events,
                )
            assert offset == len(predictions_cpu)
            return

        assert len(predictions_cpu) >= batch.size
        for index, req in enumerate(batch.reqs):
            if isinstance(req, ChunkedReq):
                continue
            self._append(
                req.uid,
                [int(predictions_cpu[index].item())],
                [float(logprobs_cpu[index].item())],
                [int(runner_up_ids_cpu[index].item())],
                [float(top2_margins_cpu[index].item())],
                [
                    {
                        "phase": batch.phase,
                        "event_start_position": len(self.token_ids.get(req.uid, [])),
                        "row_offset": 0,
                    }
                ],
            )

    def align(
        self, uid: int, output_ids: list[int]
    ) -> tuple[list[float], list[int], list[float], list[dict[str, Any]]]:
        traced_ids = self.token_ids.get(uid, [])
        traced_logprobs = self.logprobs.get(uid, [])
        runner_up_ids = self.runner_up_ids.get(uid, [])
        top2_margins = self.top2_margins.get(uid, [])
        token_events = self.token_events.get(uid, [])
        if (
            len(traced_ids) == len(output_ids) + 1
            and traced_ids[:-1] == output_ids
            and traced_ids[-1] == self.eos_token_id
        ):
            traced_ids = traced_ids[:-1]
            traced_logprobs = traced_logprobs[:-1]
            runner_up_ids = runner_up_ids[:-1]
            top2_margins = top2_margins[:-1]
            token_events = token_events[:-1]
        assert traced_ids == output_ids
        assert (
            len(traced_logprobs)
            == len(runner_up_ids)
            == len(top2_margins)
            == len(token_events)
            == len(output_ids)
        )
        return traced_logprobs, runner_up_ids, top2_margins, token_events


def _tokenize_articles(
    tokenizer: Any, articles: list[str], max_input_tokens: int
) -> list[list[int]]:
    prefix = "Summarize the following news article in 3-4 sentences:\n\n"
    suffix = "\n\nSummary:"
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    tokenized: list[list[int]] = []
    for article in articles:
        prefix_and_article = tokenizer.encode(prefix + article, add_special_tokens=True)
        keep = max_input_tokens - len(suffix_ids)
        assert keep > 0
        tokenized.append(prefix_and_article[:keep] + suffix_ids)
    return tokenized


def _worker(args: argparse.Namespace) -> None:
    import torch

    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    cases = json.loads(Path(args.cases).read_text())
    speculative = args.worker == "speculative"
    model = os.environ.get("MINISGL_CNN_MODEL", "Qwen/Qwen3-0.6B")
    attention_backend = os.environ.get("MINISGL_ATTENTION_BACKEND", "fa")
    ngram_size = int(os.environ.get("MINISGL_NGRAM_SIZE", DEFAULT_NGRAM_SIZE))
    num_draft_tokens = int(
        os.environ.get("MINISGL_NUM_DRAFT_TOKENS", DEFAULT_NUM_DRAFT_TOKENS)
    )
    spec_decoding_kwargs = (
        {
            "spec_decoding": "ngram",
            "spec_decoding_config": json.dumps(
                {
                    "ngram_size": ngram_size,
                    "num_draft_tokens": num_draft_tokens,
                }
            ),
        }
        if speculative
        else {}
    )
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
        **spec_decoding_kwargs,
    )
    trace = _ForwardTrace(llm.eos_token_id)
    original_forward = llm.engine.model.forward

    def traced_forward() -> Any:
        logits = original_forward()
        trace.record(llm.engine.ctx.batch, logits)
        return logits

    llm.engine.model.forward = traced_forward
    all_token_ids: list[list[int]] = []
    all_logprobs: list[list[float]] = []
    all_runner_up_ids: list[list[int]] = []
    all_top2_margins: list[list[float]] = []
    all_token_events: list[list[dict[str, Any]]] = []
    try:
        for start in range(0, len(cases), args.batch_size):
            batch_cases = cases[start : start + args.batch_size]
            prompt_ids = _tokenize_articles(
                llm.tokenizer,
                [case["article"] for case in batch_cases],
                args.max_input_tokens,
            )
            trace.reset()
            results = llm.generate(
                prompt_ids,
                SamplingParams(
                    temperature=0.0,
                    ignore_eos=args.ignore_eos,
                    max_tokens=args.max_output_tokens,
                ),
            )
            for uid, result in enumerate(results):
                output_ids = result["token_ids"]
                assert isinstance(output_ids, list)
                all_token_ids.append(output_ids)
                logprobs, runner_up_ids, top2_margins, token_events = trace.align(
                    uid, output_ids
                )
                all_logprobs.append(logprobs)
                all_runner_up_ids.append(runner_up_ids)
                all_top2_margins.append(top2_margins)
                all_token_events.append(token_events)

        stats = llm.speculator.stats if llm.speculator is not None else None
        speculative_stats = {
            "lookup_attempts": stats.lookup_attempts if stats else 0,
            "lookup_matches": stats.lookup_matches if stats else 0,
            "verify_steps": stats.verify_steps if stats else 0,
            "drafted_tokens": stats.drafted_tokens if stats else 0,
            "accepted_drafts": stats.accepted_drafts if stats else 0,
        }
    finally:
        llm.shutdown()

    Path(args.result).write_text(
        json.dumps(
            {
                "mode": args.worker,
                "max_output_tokens": args.max_output_tokens,
                "ignore_eos": args.ignore_eos,
                "ngram_size": ngram_size if speculative else 0,
                "num_draft_tokens": num_draft_tokens if speculative else 0,
                "case_ids": [case["id"] for case in cases],
                "token_ids": all_token_ids,
                "output_lengths": [len(token_ids) for token_ids in all_token_ids],
                "logprobs": all_logprobs,
                "runner_up_ids": all_runner_up_ids,
                "top2_margins": all_top2_margins,
                "token_events": all_token_events,
                "speculative_stats": speculative_stats,
                "gpu": torch.cuda.get_device_name(),
            }
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("baseline", "speculative"))
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    parser.add_argument(
        "--max-output-tokens", type=int, default=DEFAULT_FINISH_MAX_OUTPUT_TOKENS
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--ignore-eos", type=_bool_arg, default=DEFAULT_IGNORE_EOS)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = _parse_args()
    if parsed.worker is None or parsed.cases is None or parsed.result is None:
        raise SystemExit("--worker, --cases, and --result are required")
    _worker(parsed)
