from __future__ import annotations

import json
import os
from typing import Any

import pytest

RUN_TEST = os.environ.get("MINISGL_RUN_NGRAM_CUDA_GRAPH_SMOKE") == "1"


@pytest.mark.skipif(
    not RUN_TEST,
    reason="set MINISGL_RUN_NGRAM_CUDA_GRAPH_SMOKE=1 to run the H100 test",
)
def test_ngram_verification_cuda_graph_smoke() -> None:
    import torch
    from minisgl.core import Batch, Req, SamplingParams
    from minisgl.env import ENV
    from minisgl.llm import LLM
    from minisgl.scheduler.speculative import NgramSpeculator

    ENV.DISABLE_OVERLAP_SCHEDULING.value = True
    model_path = os.environ.get("MINISGL_CUDA_GRAPH_MODEL", "Qwen/Qwen3-0.6B")
    llm = LLM(
        model_path,
        attention_backend=os.environ.get("MINISGL_ATTENTION_BACKEND", "fi"),
        cache_type="radix",
        cuda_graph_bs=[1],
        max_extend_tokens=512,
        max_running_req=2,
        max_seq_len_override=512,
        num_page_override=2048,
        page_size=1,
        spec_decoding="ngram",
        spec_decoding_config=json.dumps({"ngram_size": 1, "num_draft_tokens": 2}),
    )
    assert isinstance(llm.speculator, NgramSpeculator)
    speculator = llm.speculator

    # After prefill, max_tokens=3 leaves two output slots. Force one real
    # proposal so verification must pad from two meaningful rows to width 3.
    def deterministic_draft(req: Req) -> torch.Tensor:
        if req.remain_len <= 1:
            return torch.empty(0, dtype=req.input_ids.dtype)
        draft_len = min(speculator.num_draft_tokens, req.remain_len - 1)
        speculator.stats.record_lookup(matched=True)
        return torch.zeros(draft_len, dtype=req.input_ids.dtype)

    speculator._draft = deterministic_draft  # type: ignore[method-assign]
    replayed_verification_shapes: list[tuple[int, int]] = []
    original_replay = llm.engine.graph_runner.replay
    original_eos_token_id = llm.eos_token_id
    llm.eos_token_id = -1

    def traced_replay(batch: Batch) -> Any:
        if batch.is_verify:
            replayed_verification_shapes.append(
                (batch.verification_len(0), batch.forward_extend_len(0))
            )
        return original_replay(batch)

    llm.engine.graph_runner.replay = traced_replay  # type: ignore[method-assign]

    try:
        outputs = llm.generate(
            ["Repeat this short test."],
            SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=3),
        )

        assert len(outputs[0]["token_ids"]) == 3
        assert replayed_verification_shapes == [(2, 3)]
        llm.cache_manager.check_integrity()
    finally:
        llm.eos_token_id = original_eos_token_id
        llm.shutdown()
