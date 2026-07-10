# Reproducible Modal performance matrix

`run_qwen_cnn_matrix_modal.py` launches Mini-SGLang on one Modal H100 for each
decode mode, then runs one concurrent CNN/DailyMail batch at concurrency 8, 16,
24, 32, 40, 48, and 64. The modes are baseline decoding, n-gram N=3/K=2, and
n-gram N=3/K=3.

The defaults use Qwen3-8B, FlashInfer, `page_size=1`, CUDA graphs disabled,
overlap scheduling disabled, greedy decoding, a 768-token post-template input
cap, a 256-token output cap, and EOS handling. All modes use the same pinned CNN
dataset, seed, warmup policy, and request selection.

Run from the repository root:

```bash
modal profile activate sandbox10
modal run --env worktrials \
  benchmark/modular-reproducible/run_qwen_cnn_matrix_modal.py
```

Use `--output-dir` or `--model` to override the local result directory or model.
Raw server logs, client logs, outputs, and summaries are persisted under the
`mini-sglang-cache` Modal Volume. The local result directory receives:

- `matrix.json`: configuration and structured metrics;
- `matrix.csv`: compact table for analysis;
- `decode_throughput_vs_tpot.svg`: P/D decode-throughput versus mean request
  TPOT, with points labeled by concurrency.

The dependency image definition intentionally matches the workspace Modal skill.
Do not modify its dependency-install commands for an individual run; doing so
invalidates the cached CUDA/PyTorch layers.
