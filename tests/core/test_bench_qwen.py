from argparse import Namespace
from pathlib import Path

import pytest


class _WhitespaceTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[str]:
        assert not add_special_tokens
        return text.split()


def test_cnn_results_separate_prefill_and_decode_metrics(tmp_path: Path) -> None:
    pytest.importorskip("openai")
    pytest.importorskip("transformers")

    from minisgl.benchmark.client import RawResult

    from benchmark.online.bench_qwen import _write_cnn_results

    args = Namespace(
        dataset_path=Path("cnn"),
        seed=0,
        port=1919,
        warmup_requests=0,
        max_tokens=4,
        ignore_eos=False,
    )
    requests = [
        {"source_index": 1, "article_id": "a", "input_len": 10},
        {"source_index": 2, "article_id": "b", "input_len": 20},
    ]
    results = [
        RawResult(
            input_len=10,
            output_len=4,
            message="a",
            tics=[0.0, 1.0, 2.0, 3.0],
            output_text="one two three four",
            output_chunks=["one", " two", " three four"],
        ),
        RawResult(
            input_len=20,
            output_len=3,
            message="b",
            tics=[0.5, 1.5, 2.5],
            output_text="five six seven",
            output_chunks=["five", " six seven"],
        ),
    ]

    summary = _write_cnn_results(
        tmp_path,
        args=args,
        model="test-model",
        tokenizer=_WhitespaceTokenizer(),
        requests=requests,
        results=results,
    )

    assert summary["ttft_ms"]["mean"] == 1000.0
    assert summary["request_tpot_ms"]["mean"] == pytest.approx(
        ((2 / 3) + 0.5) / 2 * 1000
    )
    assert summary["inter_chunk_latency_ms"]["count"] == 3
    assert summary["inter_chunk_latency_ms"]["mean"] == 1000.0
    assert summary["effective_total_decode_throughput"] == {
        "tokens_per_second": 2.5,
        "decode_tokens": 5,
        "decode_window_seconds": 2.0,
        "formula": "sum(max(completion_tokens - 1, 0)) / "
        "(latest_last_token_time - earliest_first_token_time)",
    }
