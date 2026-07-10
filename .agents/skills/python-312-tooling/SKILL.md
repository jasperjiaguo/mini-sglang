---
name: python-312-tooling
description: Standardize Python 3.12, uv, and Modal H100 workflows for mini-sglang. Use when setting up Mini-SGLang, running its models locally or on Modal, managing model/JIT caches, or installing Python tools in this workspace.
---

# Python 3.12 and Modal Mini-SGLang

Use Python 3.12 for local tooling. Use the Modal `worktrials` Environment for Linux/H100 execution; Mini-SGLang's CUDA dependencies do not run on macOS.

## Local Python

- Invoke `python3.12` outside a managed environment.
- Create project environments with `uv venv --python 3.12`.
- Install standalone CLIs with `uv tool install --python 3.12 <package>`.
- Do not change the machine-wide `python` or `python3` defaults.
- Do not use `sudo pip` or globally install project packages.

## Modal H100

- Use the authenticated `sandbox10` profile and `worktrials` Environment.
- Request `gpu="H100!"` to require an H100 rather than allow Modal to substitute an H200.
- Use TP=1, greedy decoding, `page_size=1`, and one attention backend for the initial task validation.
- Use `attention_backend="fi"` for current smoke tests. The current FlashAttention path has a CUTLASS/sgl-kernel dependency mismatch.

## Qwen3 Chat Template and Thinking

- Render chat prompts with the tokenizer bundled with the exact model revision;
  do not hand-maintain Qwen control tokens or `<think>` placement.
- For switchable Qwen3 models such as `Qwen/Qwen3-0.6B` and
  `Qwen/Qwen3-8B`, pass `enable_thinking` to `apply_chat_template` explicitly:

  ```python
  messages = [{"role": "user", "content": prompt}]
  rendered_prompt = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True,
      enable_thinking=True,
  )
  ```

- Set `enable_thinking=True` to allow a generated `<think>...</think>` block;
  it is also the default for the original switchable Qwen3 models. Set
  `enable_thinking=False` to hard-disable thinking. When the hard switch is
  enabled, `/think` and `/no_think` in the latest user or system message are
  optional per-turn soft switches.
- For `Qwen/Qwen3-0.6B`, the thinking template ends at
  `<|im_start|>assistant\n` and the model generates `<think>` itself. With
  `enable_thinking=False`, the template instead pre-fills
  `<think>\n\n</think>\n\n`. Never append either shape manually.
- Treat fixed-mode releases according to their model card:
  `Qwen3-*-Instruct-2507` is non-thinking-only and
  `Qwen3-*-Thinking-2507` is thinking-only.
- Mini-SGLang's current `TokenizeManager` does not forward
  `chat_template_kwargs`. To guarantee the mode, pre-render with the tokenizer
  as above and submit the resulting string, rather than a message list. For an
  OpenAI-compatible server that supports the option, send
  `chat_template_kwargs: {"enable_thinking": true}` in the request extension.
- Qwen recommends sampled decoding for reasoning quality. The current n-gram
  correctness benchmark uses `temperature=0` only to obtain deterministic,
  token-identical baseline and speculative runs; do not present that setting
  as a Qwen reasoning-quality evaluation.

## Persistent Models and JIT Caches

- Persist all reusable model and compilation data in the Modal Volume named `mini-sglang-cache`.
- Mount it at `/mnt/mini-sglang-cache`; do not mount it at `/root/.cache` or `/cache`, which can be non-empty in Modal images.
- Set cache environment variables at function runtime, after mounting the Volume:

  ```python
  HF_HOME = "/mnt/mini-sglang-cache/huggingface"
  XDG_CACHE_HOME = "/mnt/mini-sglang-cache"
  FLASHINFER_WORKSPACE_BASE = "/mnt/mini-sglang-cache/flashinfer"
  TVM_FFI_CACHE_DIR = "/mnt/mini-sglang-cache/tvm-ffi"
  TORCH_EXTENSIONS_DIR = "/mnt/mini-sglang-cache/torch_extensions"
  ```

- Call `volume.commit()` after downloading a model or producing JIT artifacts.
- Qwen snapshots live under `/mnt/mini-sglang-cache/huggingface/hub/models--Qwen--.../snapshots/`.

## Image Requirements

- Base the inference image on `nvidia/cuda:12.8.1-devel-ubuntu22.04` with Python 3.12. The `devel` image supplies `nvcc` for Mini-SGLang's JIT kernels.
- Install `git`, `libnuma1`, and `uv`, then run the repository setup:

  ```bash
  git clone --depth 1 https://github.com/sgl-project/mini-sglang.git /root/mini-sglang
  cd /root/mini-sglang
  uv venv --python=3.12
  . .venv/bin/activate
  uv pip install -e .
  ```

- When starting Mini-SGLang via its venv interpreter, prepend `/root/mini-sglang/.venv/bin` to `PATH` so JIT compilation can find `ninja`.

## Verified State

- `Qwen/Qwen3-0.6B` runs successfully on the H100 through Mini-SGLang with FlashInfer and `page_size=1`.
- `Qwen/Qwen3-8B` is the final verification target; download it to the same Volume before running the H100 smoke test.
