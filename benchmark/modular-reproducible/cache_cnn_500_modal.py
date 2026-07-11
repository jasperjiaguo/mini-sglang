from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import modal

APP = modal.App("mini-sglang-cache-cnn-500")
CACHE_ROOT = "/mnt/mini-sglang-cache"
HF_HOME = f"{CACHE_ROOT}/huggingface"
DATASET_PATH = f"{CACHE_ROOT}/datasets/cnn_dailymail-3.0.0-test-500-seed-0"
DATASET_REPO = "abisee/cnn_dailymail"
DATASET_CONFIG = "3.0.0"
DATASET_REVISION = "96df5e686bee6baa90b8bee7c28b81fa3fa6223d"
ROWS = 500

CACHE = modal.Volume.from_name("mini-sglang-cache", environment_name="worktrials")
IMAGE = modal.Image.from_name(
    "mini-sglang-benchmark-cu128-py312:v1",
    environment_name="worktrials",
)


@APP.function(image=IMAGE, timeout=60 * 60, volumes={CACHE_ROOT: CACHE})
def cache_dataset() -> dict[str, object]:
    os.environ["HF_HOME"] = HF_HOME
    os.environ["XDG_CACHE_HOME"] = CACHE_ROOT
    sys.path.insert(0, "/root/mini-sglang/.venv/lib/python3.12/site-packages")

    from datasets import load_dataset, load_from_disk

    dataset = load_dataset(
        DATASET_REPO,
        DATASET_CONFIG,
        split="test",
        revision=DATASET_REVISION,
        cache_dir=f"{HF_HOME}/datasets",
    )
    indices = random.Random(0).sample(range(len(dataset)), ROWS)
    expected_ids = [dataset[index]["id"] for index in indices]
    destination = Path(DATASET_PATH)
    if (destination / "state.json").exists():
        subset = load_from_disk(DATASET_PATH)
        if len(subset) != ROWS or subset["id"] != expected_ids:
            raise RuntimeError(f"existing subset does not match pinned selection: {DATASET_PATH}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        subset = dataset.select(indices)
        subset.save_to_disk(DATASET_PATH)

    manifest = {
        "dataset": DATASET_REPO,
        "config": DATASET_CONFIG,
        "revision": DATASET_REVISION,
        "source_split": "test",
        "source_rows": len(dataset),
        "selection": {"method": "random.sample", "seed": 0, "rows": ROWS},
        "indices": indices,
        "ids": expected_ids,
    }
    (destination / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    CACHE.commit()
    return {"dataset_path": DATASET_PATH, "rows": len(subset), "revision": DATASET_REVISION}


@APP.local_entrypoint()
def main() -> None:
    print(json.dumps(cache_dataset.remote(), indent=2))
