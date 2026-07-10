from __future__ import annotations

import json
import os
from typing import Any

import pytest


RUN_TEST = os.environ.get("MINISGL_RUN_NGRAM_SERVER_CORRECTNESS") == "1"


@pytest.mark.skipif(
    not RUN_TEST,
    reason="set MINISGL_RUN_NGRAM_SERVER_CORRECTNESS=1 to run the H100 test",
)
def test_ngram_server_correctness_matrix() -> None:
    import torch

    from minisgl.core import Batch, Req, SamplingParams
    from minisgl.env import ENV
    from minisgl.llm import LLM
    from minisgl.message import AbortBackendMsg
    from minisgl.scheduler.speculative import NgramSpeculator, VerificationResult

    ENV.DISABLE_OVERLAP_SCHEDULING.value = True
    model_path = os.environ.get("MINISGL_CORRECTNESS_MODEL", "Qwen/Qwen3-0.6B")
    llm = LLM(
        model_path,
        attention_backend=os.environ.get("MINISGL_ATTENTION_BACKEND", "fi"),
        cache_type="radix",
        cuda_graph_max_bs=0,
        max_extend_tokens=2048,
        max_running_req=8,
        max_seq_len_override=512,
        num_page_override=4096,
        page_size=1,
        spec_decoding="ngram",
        spec_decoding_config=json.dumps(
            {"ngram_size": 1, "num_draft_tokens": 2}
        ),
    )
    assert isinstance(llm.speculator, NgramSpeculator)
    speculator = llm.speculator

    # N-gram lookup itself has CPU unit coverage. Force a bounded linear draft
    # here so every run deterministically exercises the real verification
    # forward, attention backend, KV writes, and scheduler reconciliation.
    def deterministic_draft(req: Req) -> torch.Tensor:
        if not req.sampling_params.is_greedy or req.remain_len <= 1:
            return torch.empty(0, dtype=req.input_ids.dtype)
        draft_len = min(speculator.num_draft_tokens, req.remain_len - 1)
        speculator.stats.record_lookup(matched=True)
        return torch.zeros(draft_len, dtype=req.input_ids.dtype)

    speculator._draft = deterministic_draft  # type: ignore[method-assign]

    engine_phases: list[str] = []
    original_forward_batch = llm.engine.forward_batch

    def traced_forward_batch(batch: Batch, args: Any) -> Any:
        engine_phases.append(batch.phase)
        return original_forward_batch(batch, args)

    llm.engine.forward_batch = traced_forward_batch  # type: ignore[method-assign]
    original_eos_token_id = llm.eos_token_id

    try:
        # Concurrent greedy requests share a verification batch. The sampled
        # request must stay on ordinary decode, and every request must respect
        # its independent max_tokens bound.
        verified_uids: list[int] = []
        verification_batch_sizes: list[int] = []
        original_verify = speculator.verify

        def traced_verify(
            batch: Batch, index: int, predictions: torch.Tensor
        ) -> VerificationResult:
            if index == 0:
                verification_batch_sizes.append(len(batch.reqs))
            verified_uids.append(batch.reqs[index].uid)
            return original_verify(batch, index, predictions)

        speculator.verify = traced_verify  # type: ignore[method-assign]
        prompts = [
            "Alpha beta alpha beta. Continue briefly.",
            "Write one short sentence about deterministic testing.",
            "One two one two. Continue briefly.",
            "State the number one.",
        ]
        max_tokens = [4, 4, 4, 1]
        params = [
            SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=4),
            SamplingParams(temperature=0.7, top_k=8, ignore_eos=True, max_tokens=4),
            SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=4),
            SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=1),
        ]
        torch.manual_seed(1234)
        outputs = llm.generate(prompts, params)

        output_lengths = [len(output["token_ids"]) for output in outputs]
        assert output_lengths == max_tokens
        assert "verify" in engine_phases
        assert any(size >= 2 for size in verification_batch_sizes)
        assert 0 in verified_uids and 2 in verified_uids
        assert 1 not in verified_uids  # sampled request bypasses speculation
        assert 3 not in verified_uids  # max_tokens=1 cannot reserve a draft + bonus
        llm.cache_manager.check_integrity()

        # A repeated prompt must reuse the real radix prefix cache while the
        # speculative strategy remains enabled.
        match_lengths: list[int] = []
        original_match_req = llm.cache_manager.match_req

        def traced_match_req(req: Any) -> Any:
            result = original_match_req(req)
            match_lengths.append(result.cuda_handle.cached_len)
            return result

        llm.cache_manager.match_req = traced_match_req  # type: ignore[method-assign]
        llm.generate(
            [prompts[0]],
            SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=1),
        )
        assert match_lengths and max(match_lengths) > 0
        llm.cache_manager.check_integrity()

        # Force the first token returned by a real verification forward to be
        # treated as EOS. This deterministically tests EOS truncation and
        # rejected-suffix cleanup without relying on a model-specific prompt.
        speculator.verify = original_verify  # type: ignore[method-assign]
        eos_verifications = 0

        def verify_with_first_token_as_eos(
            batch: Batch, index: int, predictions: torch.Tensor
        ) -> VerificationResult:
            nonlocal eos_verifications
            result = original_verify(batch, index, predictions)
            eos_verifications += 1
            batch.reqs[index].sampling_params.ignore_eos = False
            llm.eos_token_id = int(result.token_ids[0].item())
            return result

        speculator.verify = verify_with_first_token_as_eos  # type: ignore[method-assign]
        eos_outputs = llm.generate(
            ["Gamma delta gamma delta. Continue briefly."],
            SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=4),
        )
        assert eos_verifications > 0
        assert len(eos_outputs[0]["token_ids"]) < 4
        llm.eos_token_id = original_eos_token_id
        speculator.verify = original_verify  # type: ignore[method-assign]
        llm.cache_manager.check_integrity()

        # Run a real prefill forward, then abort the live decode request through
        # the scheduler handler and verify both table and radix resources return.
        available_tables = llm.table_manager.available_size
        llm.pending_requests = [
            (
                "Abort this request after its prefill forward.",
                SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=8),
            )
        ]
        llm.status_map = {}
        llm.counter = 0
        with llm.engine_stream_ctx:
            llm.engine.stream.wait_stream(llm.stream)
            llm.normal_loop()
            assert llm.decode_manager.runnable
            llm._process_one_msg(AbortBackendMsg(uid=0))

        assert not llm.decode_manager.runnable
        assert not llm.prefill_manager.runnable
        assert llm.table_manager.available_size == available_tables
        llm.cache_manager.check_integrity()
    finally:
        llm.eos_token_id = original_eos_token_id
        llm.shutdown()
