# N-gram speculative decoding

This branch implements the first n-gram speculative-decoding version for
Mini-SGLang. It is intentionally a narrow, correctness-first implementation:
a request drafts one continuation chain from its own token history, verifies
that chain with the target model, then keeps only the accepted KV-cache prefix.

## Enable it

Speculation is disabled by default. Enable the n-gram strategy and pass its
configuration as JSON:

```bash
MINISGL_DISABLE_OVERLAP_SCHEDULING=1 \
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
`ngram_size=3` and `num_draft_tokens=4`, the scheduler finds an earlier
occurrence of the final three tokens and drafts up to four tokens that followed
that occurrence.

The current selection policy is **most recent matching occurrence**. A missing
match falls back to ordinary one-token decode with no speculative KV allocation.

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

## Metrics

At server shutdown, the scheduler logs:

- n-gram lookup attempts, matches, misses, and match rate;
- verification steps, drafted tokens, accepted drafts, and mean accepted
  drafts per verification step;
- conditional acceptance for each draft position, such as `p0=30/42` and
  `p1=18/30`. Position `p1` is measured only after `p0` was accepted.

Sampled requests and requests with at most one output token remaining bypass
n-gram lookup, so they are excluded from lookup statistics.

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

## Current constraints

- Tensor parallelism must be `1`.
- `--page-size` must be `1`.
- The attention backend must be FlashAttention: `--attn fa`.
- `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` is required.
- Only greedy requests speculate. Temperature-sampled requests continue with
  ordinary decode.
- CUDA graphs are used for ordinary decode but not for variable-length
  verification steps.

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
`MINISGL_CNN_MAX_OUTPUT_TOKENS` only as a safety cap.

The preserved 2000-case artifact below is the earlier fixed-length run:
`ignore_eos=true`, `max_output_tokens=32`, FI backend, batch size `8`, `n=3`,
and `k=4`. On commit `c4461f6`, it reported `385 / 2000` token-mismatched
requests (`19.25%`).

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
3. Re-enable overlap scheduling safely for speculative requests.
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
