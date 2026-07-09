from __future__ import annotations

import json
import os
from pathlib import Path

import modal


APP_NAME = "mini-sglang-cache-humaneval"
CACHE_MOUNT = "/mnt/mini-sglang-cache"
HF_HOME = f"{CACHE_MOUNT}/huggingface"
DATASET_DIR = f"{CACHE_MOUNT}/datasets/openai_humaneval-test-164"
MANIFEST_PATH = f"{DATASET_DIR}/benchmark_manifest.json"
DATASET_REPO = "openai/openai_humaneval"
DATASET_REVISION = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
EXPECTED_ROWS = 164

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("mini-sglang-cache", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install("datasets>=3,<5")


@app.function(
    image=image,
    volumes={CACHE_MOUNT: cache_volume},
    timeout=60 * 60,
)
def cache_dataset() -> str:
    os.environ["HF_HOME"] = HF_HOME
    os.environ["XDG_CACHE_HOME"] = CACHE_MOUNT

    from datasets import load_dataset, load_from_disk

    dataset = load_dataset(
        DATASET_REPO,
        split="test",
        revision=DATASET_REVISION,
        cache_dir=f"{HF_HOME}/datasets",
    )
    expected_task_ids = [f"HumanEval/{index}" for index in range(EXPECTED_ROWS)]
    if len(dataset) != EXPECTED_ROWS or dataset["task_id"] != expected_task_ids:
        raise RuntimeError(
            "Pinned HumanEval test split does not match the expected 164 tasks"
        )

    dataset_path = Path(DATASET_DIR)
    state_path = dataset_path / "state.json"
    if state_path.exists():
        cached = load_from_disk(DATASET_DIR)
        if len(cached) != EXPECTED_ROWS or cached["task_id"] != expected_task_ids:
            raise RuntimeError(f"Existing HumanEval cache is invalid: {DATASET_DIR}")
    else:
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(DATASET_DIR)
        cached = dataset

    manifest = {
        "dataset": DATASET_REPO,
        "revision": DATASET_REVISION,
        "source_split": "test",
        "rows": len(cached),
        "task_ids": expected_task_ids,
        "columns": cached.column_names,
        "hf_cache": f"{HF_HOME}/datasets",
        "saved_dataset": DATASET_DIR,
    }
    Path(MANIFEST_PATH).write_text(json.dumps(manifest, indent=2) + "\n")
    cache_volume.commit()

    verified = load_from_disk(DATASET_DIR)
    if len(verified) != EXPECTED_ROWS or verified["task_id"] != expected_task_ids:
        raise RuntimeError("HumanEval verification failed after Volume commit")

    return json.dumps(
        {
            "status": "ok",
            "cached_rows": len(verified),
            "columns": verified.column_names,
            "first_task_id": verified[0]["task_id"],
            "last_task_id": verified[-1]["task_id"],
            "dataset_path": DATASET_DIR,
            "manifest_path": MANIFEST_PATH,
        },
        indent=2,
    )


@app.local_entrypoint()
def main() -> None:
    print(cache_dataset.remote())
