from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
from pathlib import Path
from typing import Any

from minisgl.benchmark.client import (
    RawResult,
    benchmark_one_batch,
    benchmark_trace,
    get_model_name,
    process_benchmark_results,
    read_qwen_trace,
    scale_traces,
)
from minisgl.utils import init_logger
from openai import AsyncOpenAI as OpenAI
from transformers import AutoTokenizer

logger = init_logger(__name__)

URL = "https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/refs/heads/main/qwen_traceA_blksz_16.jsonl"
CNN_DATASET_REVISION = "96df5e686bee6baa90b8bee7c28b81fa3fa6223d"
CNN_SYSTEM_PROMPT = (
    "You are a careful news editor. Summarize only facts stated in the supplied article."
)
CNN_USER_PREFIX = (
    "Summarize the following CNN/DailyMail article in exactly three concise bullet points. "
    "Do not add facts or commentary.\n\nARTICLE:\n"
)


def download_qwen_trace(url: str) -> str:
    directory = Path(os.path.dirname(__file__))
    file_path = directory / "qwen_traceA_blksz_16.jsonl"
    if not file_path.exists():
        import urllib.request

        logger.info(f"Downloading trace from {url} to {file_path}...")
        urllib.request.urlretrieve(url, file_path)
        logger.info("Download completed.")
    return str(file_path)


def _chat_template_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return list(ids)


def _cnn_messages(tokenizer: Any, article: str) -> tuple[list[dict[str, str]], int]:
    messages = [
        {"role": "system", "content": CNN_SYSTEM_PROMPT},
        {"role": "user", "content": CNN_USER_PREFIX + article},
    ]
    input_len = len(_chat_template_ids(tokenizer, messages))
    return messages, input_len


def _load_cnn_requests(
    dataset_path: Path,
    tokenizer: Any,
    *,
    num_requests: int,
    warmup_requests: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from datasets import load_from_disk

    dataset = load_from_disk(str(dataset_path))
    total = num_requests + warmup_requests
    if total > len(dataset):
        raise ValueError(f"requested {total} CNN rows, but dataset contains {len(dataset)}")
    indices = random.Random(seed).sample(range(len(dataset)), total)
    measured_indices = indices[:num_requests]
    warmup_indices = indices[num_requests:]
    requests: list[dict[str, Any]] = []
    for source_index in warmup_indices + measured_indices:
        row = dataset[source_index]
        messages, input_len = _cnn_messages(tokenizer, row["article"])
        requests.append(
            {
                "source_index": source_index,
                "article_id": row["id"],
                "messages": messages,
                "input_len": input_len,
            }
        )
    return requests[:warmup_requests], requests[warmup_requests:]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


def _latency_stats_ms(values_seconds: list[float]) -> dict[str, float | int]:
    if not values_seconds:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
    values_ms = [value * 1000 for value in values_seconds]
    return {
        "count": len(values_ms),
        "mean": statistics.fmean(values_ms),
        "p50": _percentile(values_ms, 0.50),
        "p90": _percentile(values_ms, 0.90),
        "p99": _percentile(values_ms, 0.99),
    }


def _write_cnn_results(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    model: str,
    tokenizer: Any,
    requests: list[dict[str, Any]],
    results: list[RawResult],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    min_start = min(result.tics[0] for result in results)
    max_end = max(result.tics[-1] for result in results)
    duration = max_end - min_start
    completion_lengths = [
        len(tokenizer.encode(result.output_text, add_special_tokens=False)) for result in results
    ]
    ttfts = [result.tics[1] - result.tics[0] for result in results]
    e2e_seconds = [result.tics[-1] - result.tics[0] for result in results]
    decode_tokens = [max(length - 1, 0) for length in completion_lengths]
    decode_seconds = [result.tics[-1] - result.tics[1] for result in results]
    request_tpots = [
        seconds / tokens
        for seconds, tokens in zip(decode_seconds, decode_tokens, strict=True)
        if tokens > 0
    ]
    inter_chunk_latencies = [
        end - start
        for result in results
        for start, end in zip(result.tics[1:-1], result.tics[2:], strict=True)
    ]
    decode_window_seconds = max(result.tics[-1] for result in results) - min(
        result.tics[1] for result in results
    )
    total_decode_tokens = sum(decode_tokens)
    effective_decode_throughput = (
        total_decode_tokens / decode_window_seconds if decode_window_seconds > 0 else 0.0
    )
    active_decode_seconds = [
        seconds
        for seconds, tokens in zip(decode_seconds, decode_tokens, strict=True)
        if tokens > 0
    ]
    mean_request_decode_seconds = (
        statistics.fmean(active_decode_seconds) if active_decode_seconds else 0.0
    )
    pd_disagg_decode_throughput = (
        total_decode_tokens / mean_request_decode_seconds
        if mean_request_decode_seconds > 0
        else 0.0
    )
    summary = {
        "workload": "cnn_dailymail_summarization",
        "dataset_path": str(args.dataset_path),
        "dataset_revision": CNN_DATASET_REVISION,
        "seed": args.seed,
        "selection": "random.sample; measured rows are selected before disjoint warmup rows",
        "model": model,
        "port": args.port,
        "num_requests": len(results),
        "warmup_requests": args.warmup_requests,
        "input_truncation": None,
        "max_tokens": args.max_tokens,
        "ignore_eos": args.ignore_eos,
        "chat_template_kwargs": {"enable_thinking": False},
        "duration_seconds": duration,
        "request_throughput_qps": len(results) / duration,
        "output_token_throughput": sum(completion_lengths) / duration,
        "completion_tokens": {
            "total": sum(completion_lengths),
            "min": min(completion_lengths),
            "mean": statistics.fmean(completion_lengths),
            "max": max(completion_lengths),
            "hit_max_tokens": sum(length >= args.max_tokens for length in completion_lengths),
        },
        "ttft_ms": _latency_stats_ms(ttfts),
        "request_tpot_ms": _latency_stats_ms(request_tpots),
        "inter_chunk_latency_ms": _latency_stats_ms(inter_chunk_latencies),
        "effective_total_decode_throughput": {
            "tokens_per_second": effective_decode_throughput,
            "decode_tokens": total_decode_tokens,
            "decode_window_seconds": decode_window_seconds,
            "formula": "sum(max(completion_tokens - 1, 0)) / "
            "(latest_last_token_time - earliest_first_token_time)",
        },
        "pd_disagg_decode_throughput": {
            "tokens_per_second": pd_disagg_decode_throughput,
            "decode_tokens": total_decode_tokens,
            "mean_request_decode_seconds": mean_request_decode_seconds,
            "concurrent_requests_with_decode_tokens": len(active_decode_seconds),
            "formula": "sum(max(completion_tokens - 1, 0)) / "
            "mean(last_token_time - first_token_time)",
        },
        "e2e_seconds": {
            "mean": statistics.fmean(e2e_seconds),
            "p50": _percentile(e2e_seconds, 0.50),
            "p90": _percentile(e2e_seconds, 0.90),
            "p99": _percentile(e2e_seconds, 0.99),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "outputs.jsonl").open("w") as output_file:
        for request, result, completion_len in zip(
            requests, results, completion_lengths, strict=True
        ):
            output_file.write(
                json.dumps(
                    {
                        "source_index": request["source_index"],
                        "article_id": request["article_id"],
                        "input_tokens": request["input_len"],
                        "completion_tokens": completion_len,
                        "ttft_ms": (result.tics[1] - result.tics[0]) * 1000,
                        "tpot_ms": (
                            (result.tics[-1] - result.tics[1])
                            / (completion_len - 1)
                            * 1000
                            if completion_len > 1
                            else None
                        ),
                        "inter_chunk_latencies_ms": [
                            (end - start) * 1000
                            for start, end in zip(
                                result.tics[1:-1], result.tics[2:], strict=True
                            )
                        ],
                        "e2e_seconds": result.tics[-1] - result.tics[0],
                        "completion": result.output_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    logger.info("CNN benchmark summary:\n%s", json.dumps(summary, indent=2))
    return summary


async def _run_cnn(args: argparse.Namespace, client: OpenAI, model: str, tokenizer: Any) -> None:
    warmup, requests = _load_cnn_requests(
        args.dataset_path,
        tokenizer,
        num_requests=args.num_requests,
        warmup_requests=args.warmup_requests,
        seed=args.seed,
    )
    extra_body = {
        "ignore_eos": args.ignore_eos,
        "top_k": 1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if warmup:
        logger.info("Warming up with %d requests", len(warmup))
        await benchmark_one_batch(
            client,
            [request["messages"][-1]["content"] for request in warmup],
            args.warmup_max_tokens,
            model,
            extra_body=extra_body,
            input_lengths=[request["input_len"] for request in warmup],
            messages=[request["messages"] for request in warmup],
            pbar=False,
        )

    logger.info(
        "Starting CNN benchmark with %d requests, max_tokens=%d, ignore_eos=%s",
        len(requests),
        args.max_tokens,
        args.ignore_eos,
    )
    results = await benchmark_one_batch(
        client,
        [request["messages"][-1]["content"] for request in requests],
        args.max_tokens,
        model,
        extra_body=extra_body,
        input_lengths=[request["input_len"] for request in requests],
        messages=[request["messages"] for request in requests],
        pbar=not args.no_progress,
    )
    process_benchmark_results(results, tokenizer)
    _write_cnn_results(
        args.output_dir,
        args=args,
        model=model,
        tokenizer=tokenizer,
        requests=requests,
        results=results,
    )


async def main() -> None:
    args = _parse_args()
    random.seed(42)
    async with OpenAI(
        base_url=f"http://127.0.0.1:{args.port}/v1",
        api_key="dummy",
        timeout=args.timeout,
        max_retries=0,
    ) as client:
        model = await get_model_name(client)
        tokenizer = AutoTokenizer.from_pretrained(model)
        if args.workload == "cnn":
            await _run_cnn(args, client, model, tokenizer)
            return

        traces = read_qwen_trace(
            download_qwen_trace(URL), tokenizer, n=args.num_requests, dummy=True
        )
        logger.info(f"Start benchmarking with {args.num_requests} requests using model {model}...")
        for scale in args.scales:
            results = await benchmark_trace(client, scale_traces(traces, scale), model)
            process_benchmark_results(results, tokenizer)
        logger.info("Benchmarking completed.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=("qwen-trace", "cnn"), default="qwen-trace")
    parser.add_argument("--port", type=int, default=1919)
    parser.add_argument("--num-requests", type=int, default=1000)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7, 0.8, 1.6])
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_qwen_output"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup-requests", type=int, default=32)
    parser.add_argument("--warmup-max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.workload == "cnn" and args.dataset_path is None:
        parser.error("--dataset-path is required for --workload cnn")
    return args


if __name__ == "__main__":
    asyncio.run(main())
