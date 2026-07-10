from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import modal


APP_NAME = "mini-sglang-cache-math500"
CACHE_MOUNT = "/mnt/mini-sglang-cache"
HF_HOME = f"{CACHE_MOUNT}/huggingface"
DATASET_DIR = f"{CACHE_MOUNT}/datasets/math-500-test-500"
MANIFEST_PATH = f"{DATASET_DIR}/benchmark_manifest.json"
DATASET_REPO = "HuggingFaceH4/MATH-500"
DATASET_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
EXPECTED_ROWS = 500

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("mini-sglang-cache", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install("datasets>=3,<5")


def dataset_sha256(dataset: object) -> str:
    digest = hashlib.sha256()
    for row in dataset:
        serialized = json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(serialized.encode())
        digest.update(b"\n")
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
        split="test",
        revision=DATASET_REVISION,
        cache_dir=f"{HF_HOME}/datasets",
    )
    if len(dataset) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Pinned MATH-500 test split has {len(dataset)} rows, "
            f"expected {EXPECTED_ROWS}"
        )
    expected_sha256 = dataset_sha256(dataset)

    dataset_path = Path(DATASET_DIR)
    state_path = dataset_path / "state.json"
    if state_path.exists():
        cached = load_from_disk(DATASET_DIR)
        if len(cached) != EXPECTED_ROWS or dataset_sha256(cached) != expected_sha256:
            raise RuntimeError(f"Existing MATH-500 cache is invalid: {DATASET_DIR}")
    else:
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(DATASET_DIR)
        cached = dataset

    manifest = {
        "dataset": DATASET_REPO,
        "revision": DATASET_REVISION,
        "source_split": "test",
        "rows": len(cached),
        "columns": cached.column_names,
        "subject_counts": dict(sorted(Counter(cached["subject"]).items())),
        "level_counts": dict(
            sorted(Counter(str(level) for level in cached["level"]).items())
        ),
        "content_sha256": expected_sha256,
        "hf_cache": f"{HF_HOME}/datasets",
        "saved_dataset": DATASET_DIR,
    }
    Path(MANIFEST_PATH).write_text(json.dumps(manifest, indent=2) + "\n")
    cache_volume.commit()

    verified = load_from_disk(DATASET_DIR)
    if len(verified) != EXPECTED_ROWS or dataset_sha256(verified) != expected_sha256:
        raise RuntimeError("MATH-500 verification failed after Volume commit")

    return json.dumps(
        {
            "status": "ok",
            "cached_rows": len(verified),
            "columns": verified.column_names,
            "content_sha256": expected_sha256,
            "first_unique_id": verified[0]["unique_id"],
            "last_unique_id": verified[-1]["unique_id"],
            "dataset_path": DATASET_DIR,
            "manifest_path": MANIFEST_PATH,
        },
        indent=2,
    )


@app.local_entrypoint()
def main() -> None:
    print(cache_dataset.remote())
