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
