# N-gram speculative decoding

This branch implements the first n-gram speculative-decoding version for
Mini-SGLang. It is intentionally a narrow, correctness-first implementation:
a request drafts one continuation chain from its own token history, verifies
that chain with the target model, then keeps only the accepted KV-cache prefix.

## Enable it

Speculation is disabled by default. Enable it with both of these arguments:

```bash
MINISGL_DISABLE_OVERLAP_SCHEDULING=1 \
python -m minisgl \
  --model Qwen/Qwen3-0.6B \
  --attn fa \
  --page-size 1 \
  --speculative-ngram-size 3 \
  --speculative-num-draft-tokens 4
```

### Configuration

| Argument | Meaning | Default |
| --- | --- | --- |
| `--speculative-ngram-size N` | Number of trailing tokens used as the n-gram lookup suffix. | `0` (disabled) |
| `--speculative-num-draft-tokens K` | Maximum number of tokens drafted after a suffix match. | `0` (disabled) |

Both values must be positive to enable speculation. With `N=3` and `K=4`,
the scheduler finds an earlier occurrence of the final three tokens and drafts
up to four tokens that followed that occurrence.

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

### FI step-extend numerical replay artifacts

Future numerical debugging should use FlashInfer (`fi`) as the primary
attention backend. The current step-wise replay experiment compares:

1. FI autoregressive baseline;
2. FI n-gram speculative decode;
3. no-spec FI replay that follows the same step cadence up to the first
   mismatch, forcing captured `draft_ids` for n-gram hits and ordinary
   one-token decode for n-gram misses.

The replay compares each mismatch row's no-spec replay logits/token against
the autoregressive baseline token and the original speculative token.

Local result artifacts:

| Batch size | Output artifact | Summary |
| --- | --- | --- |
| `8` | `/private/tmp/fi_step_extend_repro_bs8_stdout.txt` | `42 / 200` mismatch cases; replay matched speculative token `37 / 42`, autoregressive token `4 / 42`, neither `1 / 42`. |
| `1` | `/private/tmp/fi_step_extend_repro_bs1_stdout.txt` | `42 / 200` mismatch cases; replay matched speculative token `42 / 42`, autoregressive token `0 / 42`. |

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
MINISGL_RUN_FI_STEP_EXTEND_REPRO=1 \
MINISGL_ATTENTION_BACKEND=fi \
MINISGL_CNN_CASES=200 \
python -m pytest -q -s -o addopts= \
  tests/integration/test_fi_step_extend_repro_numerics.py
```

The bs=1 script additionally sets `MINISGL_CNN_BATCH_SIZE=1`; the bs=8 script
uses the test default `DEFAULT_BATCH_SIZE=8`.

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
