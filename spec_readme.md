# N-gram speculative decoding

This branch implements the first n-gram speculative-decoding version for
Mini-SGLang. It is intentionally a narrow, correctness-first implementation:
a request drafts one continuation chain from its own token history, verifies
that chain with the target model, then keeps only the accepted KV-cache prefix.

## Enable it

Speculation is disabled by default. Enable the n-gram strategy and pass its
configuration as JSON:

```bash
python -m minisgl \
  --model Qwen/Qwen3-0.6B \
  --attn fa \
  --page-size 1 \
  --spec-decoding ngram \
  --spec-decoding-config '{"ngram_size": 3, "num_draft_tokens": 4}'
```

### Configuration

| Argument | Meaning | Default |
| --- | --- | --- |
| `--spec-decoding ngram` | Select the n-gram speculative strategy. | unset (disabled) |
| `--spec-decoding-config JSON` | N-gram configuration with positive `ngram_size` and `num_draft_tokens` integers. | unset |

The config is rejected unless `--spec-decoding` is also provided. With
`ngram_size=3` and `num_draft_tokens=4`, the scheduler first looks for an
earlier occurrence of the final three tokens and drafts up to four tokens that
followed that occurrence. If no three-token match exists, it retries with the
final two tokens and then the final one token.

`ngram_size` is therefore the maximum lookup length. The matcher prioritizes
the **longest matching suffix**, and within one suffix length it selects the
**most recent matching occurrence**. If no suffix from `ngram_size` through
one token matches, the request falls back to ordinary one-token decode with no
speculative KV allocation.

## Verification behavior

Every active decode request has one pending token without KV cache. A
verification step forwards:

```text
pending token + K drafted tokens
```

The target model produces `K + 1` predictions. Matching drafts are accepted
left-to-right. On the first mismatch, the target-model token replaces the
mismatched draft; if all drafts match, the extra target-model prediction is the
bonus token. In either case, the final emitted token becomes the next pending
token without KV cache.

### Verified H100 CUDA-graph configuration

The CNN/DailyMail semantic smoke test used Modal `gpu="H100!"`, the
`worktrials` environment, and the persistent `mini-sglang-cache` Volume. The
engine was started directly through `LLM` with this exact configuration:

```python
import json

from minisgl.env import ENV
from minisgl.llm import LLM

ENV.DISABLE_OVERLAP_SCHEDULING.value = True

llm = LLM(
    "Qwen/Qwen3-8B",
    attention_backend="fi",
    cache_type="radix",
    cuda_graph_bs=[4],
    max_extend_tokens=8192,
    max_running_req=8,
    max_seq_len_override=4096,
    num_page_override=12288,
    page_size=1,
    spec_decoding="ngram",
    spec_decoding_config=json.dumps(
        {"ngram_size": 1, "num_draft_tokens": 2}
    ),
)
```

`cuda_graph_bs=[4]` is a request-count bucket. With two draft tokens, each
verification request has a fixed physical width of `K + 1 = 3`, so the
verification graph forwards `4 * 3 = 12` flattened token rows. A batch with
fewer than four real requests is padded to this request bucket.

The three concurrent summarization requests used normal EOS handling and a
256-token safety cap:

```python
from minisgl.core import SamplingParams

sampling_params = SamplingParams(
    temperature=0.0,
    ignore_eos=False,
    max_tokens=256,
)
```

Qwen control tokens and thinking placement were produced by the tokenizer,
not assembled manually:

```python
messages = [
    {
        "role": "system",
        "content": (
            "You are a careful news editor. Summarize only facts stated "
            "in the supplied article."
        ),
    },
    {
        "role": "user",
        "content": (
            "Summarize the following CNN/DailyMail article in exactly three "
            "concise bullet points. Do not add facts or commentary.\n\n"
            "ARTICLE:\n" + article
        ),
    },
]
prompt = llm.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
```

The run exercised verification CUDA graphs with real n-gram matches; it did
not replace the n-gram matcher with a deterministic test draft.

## Metrics

At server shutdown, the scheduler logs:

- n-gram lookup attempts, matches, misses, match rate, and matches grouped by
  suffix length;
- verification steps, drafted tokens, accepted drafts, and mean accepted
  drafts per verification step;
- conditional acceptance for each draft position, such as `p0=30/42` and
  `p1=18/30`. Position `p1` is measured only after `p0` was accepted.

Requests with at most one output token remaining bypass n-gram lookup, so they
are excluded from lookup statistics. Greedy and temperature-sampled requests
can both speculate; sampled verification uses deterministic-proposal rejection
sampling.

## Benchmark dataset

### CNN/Dailymail

The n-gram speculative-decoding benchmark uses 100 summarization examples from
[CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail), config
`3.0.0`, test split. The source dataset is pinned to revision
`96df5e686bee6baa90b8bee7c28b81fa3fa6223d`; the benchmark subset is selected
with `random.sample(seed=0)` and saved with its source indices and article IDs.

On Modal, the full 11,490-row test split is cached in the `mini-sglang-cache`
Volume and the frozen subset is stored at:

```text
/mnt/mini-sglang-cache/datasets/cnn_dailymail-3.0.0-test-100-seed-0
```

Populate or verify the cache in the `worktrials` environment with:

```bash
modal run --env worktrials benchmark/offline/cache_cnn_dailymail_modal.py
```

CNN/DailyMail benchmark and numerical-test prompts always preserve the complete
article. They expose no input-length truncation setting; GPU tests derive their
buffer sizes from the longest tokenized prompt in the selected cases.

### HumanEval

The code-generation benchmark uses all 164 problems from the `test` split of
[OpenAI HumanEval](https://huggingface.co/datasets/openai/openai_humaneval).
The dataset is pinned to revision
`7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544` and includes `task_id`, `prompt`,
`canonical_solution`, `test`, and `entry_point`.

On Modal, the dataset and its manifest are persisted in the
`mini-sglang-cache` Volume at:

```text
/mnt/mini-sglang-cache/datasets/openai_humaneval-test-164
```

Populate or verify the cache in the `worktrials` environment with:

```bash
modal run --env worktrials benchmark/offline/cache_humaneval_modal.py
```

### GSM8K

The mathematical-reasoning benchmark uses all 1,319 problems from the `test`
split of the `main` configuration of
[GSM8K](https://huggingface.co/datasets/openai/gsm8k). The dataset is pinned to
revision `740312add88f781978c0658806c59bc2815b9866` and includes `question` and
`answer`. Its manifest records a SHA-256 digest of the complete split because
GSM8K does not provide a task-ID column.

On Modal, the dataset and its manifest are persisted in the
`mini-sglang-cache` Volume at:

```text
/mnt/mini-sglang-cache/datasets/gsm8k-main-test-1319
```

Populate or verify the cache in the `worktrials` environment with:

```bash
modal run --env worktrials benchmark/offline/cache_gsm8k_modal.py
```

### MATH-500

The hard mathematical-reasoning benchmark uses all 500 problems from the
`test` split of
[MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500). The
dataset is pinned to revision
`6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`; its manifest records subject and
difficulty-level counts plus a SHA-256 digest of every complete row. Benchmark
prompts should use only the `problem` field, keeping the reference `solution`
and `answer` fields out of the model input.

On Modal, the dataset and its manifest are persisted in the
`mini-sglang-cache` Volume at:

```text
/mnt/mini-sglang-cache/datasets/math-500-test-500
```

Populate or verify the cache in the `worktrials` environment with:

```bash
modal run --env worktrials benchmark/offline/cache_math500_modal.py
```

## Current constraints

- Tensor parallelism must be `1`.
- `--page-size` must be `1`.
- The attention backend must be FlashAttention or FlashInfer: `--attn fa` or
  `--attn fi`.
- Verification CUDA graphs use a fixed physical width of `K + 1` per request.
  Short real drafts are padded, but padding is excluded from acceptance and
  real KV allocation. Unsupported graph batch sizes and requests too close to
  the model context limit fall back to the eager variable-width path.

Overlap scheduling is supported by keeping the batch currently executing on
the engine stream out of the next speculative scheduling decision. This lets a
disjoint request group execute while the scheduler reconciles the prior
group's variable-length verification result. For a CUDA graph bucket of 32,
use at least 64 running requests to keep two full request groups available.
Chunked-prefill continuations are reconciled before their request table can be
reused.

## Misc

### FlashAttention 3 dependency compatibility

The current `sgl_kernel` FlashAttention 3 interface uses
`nvvm.RoundingModeKind`, which was removed in `nvidia-cutlass-dsl` 4.6. The
project therefore pins `nvidia-cutlass-dsl==4.5.3`. This is a dependency
compatibility requirement, not a speculative-decoding runtime constraint.

### CNN n-gram broad numerical artifact

The broad CNN/DailyMail numerics test compares normal greedy decoding against
n-gram speculative greedy decoding for full generated sequences. The test now
honors EOS by default (`MINISGL_CNN_IGNORE_EOS=0`) and uses
`MINISGL_CNN_MAX_OUTPUT_TOKENS` only as a safety cap. Current CNN numerical
tests render the article prompt with Qwen's chat-template shape and
`enable_thinking=false`:

```text
<|im_start|>user
Summarize the following news article in 3-4 sentences. Return only the summary,
without analysis or extra headings.

{article}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

The preserved 2000-case artifact below is the earlier fixed-length run:
legacy raw completion prompt, `ignore_eos=true`, `max_output_tokens=32`, FI
backend, batch size `8`, `n=3`, and `k=4`. On commit `c4461f6`, it reported
`385 / 2000` token-mismatched requests (`19.25%`).

Preserved artifact:

```text
benchmark/result/numerical/fi_ngram_2000_stdout.txt
```

The test intentionally failed because
`tests/integration/test_ngram_speculative_numerics.py` asserts exact token
equality; the JSON artifact preserves the printed summary, including logprob
deltas and speculative acceptance stats.

To run the EOS-honoring version:

```bash
MINISGL_RUN_CNN_NUMERICS=1 \
MINISGL_ATTENTION_BACKEND=fi \
MINISGL_CNN_CASES=2000 \
MINISGL_CNN_IGNORE_EOS=0 \
MINISGL_CNN_MAX_OUTPUT_TOKENS=256 \
python -m pytest -q -s -o addopts= \
  tests/integration/test_ngram_speculative_numerics.py
```

### Mini-vs-main Qwen3-0.6B bs=1 numerical artifact

This legacy raw-prompt artifact compares Mini-SGLang and upstream/main SGLang
on the same stable-seed sample of 200 CNN/DailyMail test requests. The run
used `Qwen/Qwen3-0.6B`, FlashInfer (`fi` in Mini-SGLang,
`attention_backend="flashinfer"` in main SGLang), batch size `1`,
`max_output_tokens=256`, `ignore_eos=false`, disabled overlap scheduling, and
disabled decode CUDA graphs. All outputs reached the 256-token cap; there were
no length mismatches.

| Comparison | Token mismatch requests | Length mismatches | Prefix logprob mean / p99 / max |
| --- | ---: | ---: | --- |
| mini baseline vs mini n-gram | `161 / 200` | `0` | `0.0106 / 0.0829 / 0.1769` |
| main baseline vs main n-gram | `171 / 200` | `0` | `0.0121 / 0.0861 / 0.2629` |
| mini baseline vs main baseline | `170 / 200` | `0` | `0.0135 / 0.1011 / 0.2215` |
| mini n-gram vs main n-gram | `178 / 200` | `0` | `0.0144 / 0.1041 / 0.2860` |

The Mini-SGLang and main SGLang n-gram mismatch sets overlapped on
`143 / 189` union mismatch requests (`75.7%` Jaccard overlap).

Preserved repo artifacts:

```text
benchmark/result/numerical/mini_main_qwen06_bs1_200/summary.json
benchmark/result/numerical/mini_main_qwen06_bs1_200/raw_tokens_logprobs.json
```

### Step-extend numerical replay artifacts

Future numerical debugging should use FlashInfer (`fi`) as the primary
attention backend. The replay test defaults to `fi`, uses batch size `1` by
default, and can also be pointed at FA3 with `MINISGL_STEP_EXTEND_BACKEND=fa`
or `MINISGL_ATTENTION_BACKEND=fa` (`fa3` is accepted as an alias). The current
step-wise replay experiment compares:

1. backend-selected autoregressive baseline;
2. backend-selected n-gram speculative decode;
3. no-spec replay on the same backend that follows the same step cadence up to
   the first mismatch, forcing captured `draft_ids` for n-gram hits and
   ordinary one-token decode for n-gram misses.

The replay compares each mismatch row's no-spec replay logits/token against
the autoregressive baseline token and the original speculative token.

Local result artifacts:

| Batch size | Output artifact | Summary |
| --- | --- | --- |
| `8` | `/private/tmp/fi_step_extend_repro_bs8_stdout.txt` | `42 / 200` mismatch cases; replay matched speculative token `37 / 42`, autoregressive token `4 / 42`, neither `1 / 42`. |
| `1` | `/private/tmp/fi_step_extend_repro_bs1_stdout.txt` | `42 / 200` mismatch cases; replay matched speculative token `42 / 42`, autoregressive token `0 / 42`. |

Preserved repo copies live under `benchmark/result/numerical/`:

| Batch size | Preserved artifact |
| --- | --- |
| `1` | `benchmark/result/numerical/fi_step_extend_repro_stdout.txt` |
| `8` | `benchmark/result/numerical/fi_step_extend_repro_bs8_stdout.txt` |

The bs=8 artifact was originally written as
`/private/tmp/fi_step_extend_repro_stdout.txt` and has been renamed to include
`bs8`.

Reproduction scripts:

```text
/private/tmp/modal_fi_step_extend_repro.py
/private/tmp/modal_fi_step_extend_repro_bs1.py
```

Run the bs=8 replay:

```bash
modal run --env worktrials \
  --write-result /private/tmp/fi_step_extend_repro_bs8_stdout.txt \
  /private/tmp/modal_fi_step_extend_repro.py
```

Run the bs=1 replay:

```bash
modal run --env worktrials \
  --write-result /private/tmp/fi_step_extend_repro_bs1_stdout.txt \
  /private/tmp/modal_fi_step_extend_repro_bs1.py
```

Both scripts run:

```bash
MINISGL_RUN_STEP_EXTEND_REPRO=1 \
MINISGL_STEP_EXTEND_BACKEND=fi \
MINISGL_CNN_CASES=200 \
python -m pytest -q -s -o addopts= \
  tests/integration/test_fi_step_extend_repro_numerics.py
```

The old `MINISGL_RUN_FI_STEP_EXTEND_REPRO=1` gate is still accepted for
compatibility. To run the same replay on FA3, set
`MINISGL_STEP_EXTEND_BACKEND=fa` or `MINISGL_STEP_EXTEND_BACKEND=fa3`. To
reproduce the bs=8 artifact, additionally set `MINISGL_CNN_BATCH_SIZE=8`;
otherwise the test uses its local default batch size of `1`.

## Pending work

1. **Support `page_size > 1`.** Rejected drafts can leave accepted and rejected
   tokens sharing a KV page. This requires page-aware suffix cleanup and, for
   some layouts, KV compaction before the request can continue.
2. **Add selectable drafting policies.** The current implementation is a
   recency-based chain. Add a `BFS`/recency policy and a `PROB`/frequency policy
   that chooses continuations based on observed occurrence counts. SGLang's
   production implementation generalizes this further into speculative trees;
   this branch will start with a single chain for both policies.
3. Stress overlap scheduling with aborts, chunked prefill, and mixed sampled
   requests under sustained load.
4. Dynamic `K`: reduce or skip speculation for low-acceptance requests and
   large decode batches.

## Follow-ups

1. **CUDA-graph the verification step.** Capture fixed-shape verification
   buckets to recover the CPU launch overhead currently avoided only by regular
   decode CUDA graphs.
2. **Correct temperature sampling via rejection sampling.** Supply a valid
   draft proposal distribution and apply target-model rejection sampling, rather
   than bypassing speculation for sampled requests.

## Validation status

Focused unit tests cover n-gram lookup, greedy acceptance, verification-batch
shape, output-length limiting, scheduling fairness, lookup failure accounting,
and per-position acceptance accounting. The local macOS workspace cannot run
the CUDA-dependent test environment. An H100 smoke test with Qwen3-0.6B,
`N=1`, and `K=4` completed 12 generated tokens, with 10 lookups, one match,
and one verification step.
