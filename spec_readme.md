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

## Current constraints

- Tensor parallelism must be `1`.
- `--page-size` must be `1`.
- The attention backend must be FlashAttention: `--attn fa`.
- FlashAttention 3 requires `nvidia-cutlass-dsl==4.5.3`; this is pinned in the
  project because CUTLASS DSL 4.6 removed an enum used by the current
  `sgl_kernel` FA3 interface.
- `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` is required.
- Only greedy requests speculate. Temperature-sampled requests continue with
  ordinary decode.
- CUDA graphs are used for ordinary decode but not for variable-length
  verification steps.

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
