from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import modal


APP_NAME = "mini-sglang-cache-gsm8k"
CACHE_MOUNT = "/mnt/mini-sglang-cache"
HF_HOME = f"{CACHE_MOUNT}/huggingface"
DATASET_DIR = f"{CACHE_MOUNT}/datasets/gsm8k-main-test-1319"
MANIFEST_PATH = f"{DATASET_DIR}/benchmark_manifest.json"
DATASET_REPO = "openai/gsm8k"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
DATASET_CONFIG = "main"
EXPECTED_ROWS = 1_319

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("mini-sglang-cache", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install("datasets>=3,<5")


def dataset_sha256(dataset: object) -> str:
    digest = hashlib.sha256()
    for row in dataset:
        digest.update(row["question"].encode())
        digest.update(b"\0")
        digest.update(row["answer"].encode())
        digest.update(b"\0")
    return digest.hexdigest()


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
        DATASET_CONFIG,
        split="test",
        revision=DATASET_REVISION,
        cache_dir=f"{HF_HOME}/datasets",
    )
    if len(dataset) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Pinned GSM8K test split has {len(dataset)} rows, expected {EXPECTED_ROWS}"
        )
    expected_sha256 = dataset_sha256(dataset)

    dataset_path = Path(DATASET_DIR)
    state_path = dataset_path / "state.json"
    if state_path.exists():
        cached = load_from_disk(DATASET_DIR)
        if len(cached) != EXPECTED_ROWS or dataset_sha256(cached) != expected_sha256:
            raise RuntimeError(f"Existing GSM8K cache is invalid: {DATASET_DIR}")
    else:
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(DATASET_DIR)
        cached = dataset

    manifest = {
        "dataset": DATASET_REPO,
        "revision": DATASET_REVISION,
        "config": DATASET_CONFIG,
        "source_split": "test",
        "rows": len(cached),
        "columns": cached.column_names,
        "content_sha256": expected_sha256,
        "hf_cache": f"{HF_HOME}/datasets",
        "saved_dataset": DATASET_DIR,
    }
    Path(MANIFEST_PATH).write_text(json.dumps(manifest, indent=2) + "\n")
    cache_volume.commit()

    verified = load_from_disk(DATASET_DIR)
    if len(verified) != EXPECTED_ROWS or dataset_sha256(verified) != expected_sha256:
        raise RuntimeError("GSM8K verification failed after Volume commit")

    return json.dumps(
        {
            "status": "ok",
            "cached_rows": len(verified),
            "columns": verified.column_names,
            "content_sha256": expected_sha256,
            "dataset_path": DATASET_DIR,
            "manifest_path": MANIFEST_PATH,
        },
        indent=2,
    )


@app.local_entrypoint()
def main() -> None:
    print(cache_dataset.remote())
