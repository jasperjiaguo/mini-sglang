from __future__ import annotations

import modal

APP = modal.App("mini-sglang-build-stable-benchmark-image")
IMAGE_NAME = "mini-sglang-benchmark-cu128-py312:v1"
REMOTE_ROOT = "/root/mini-sglang"
SOURCE_COMMIT = "c6369d5"

# This image is deliberately pinned and built only when its version changes.
# Routine benchmarks consume IMAGE_NAME and add the current source separately.
DEPENDENCY_IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "libnuma1")
    .pip_install("uv")
    .run_commands(
        "git clone https://github.com/jasperjiaguo/mini-sglang.git "
        f"{REMOTE_ROOT}",
        f"cd {REMOTE_ROOT} && git checkout --detach {SOURCE_COMMIT}",
        f"cd {REMOTE_ROOT} && uv venv --python=3.12",
        f"cd {REMOTE_ROOT} && . .venv/bin/activate && "
        "uv pip install -e . 'datasets>=3,<5' pytest ruff",
        f"cd {REMOTE_ROOT} && . .venv/bin/activate && "
        "python -c \"import datasets, openai, pytest, torch, transformers; "
        "import flashinfer; print(torch.__version__)\"",
    )
)


@APP.local_entrypoint()
def main(name: str = IMAGE_NAME) -> None:
    image = DEPENDENCY_IMAGE.build(APP)
    image.publish(name, environment_name="worktrials")
    print(f"Published {name} as {image.object_id}")
