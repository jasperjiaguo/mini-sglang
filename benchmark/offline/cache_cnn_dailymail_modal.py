from __future__ import annotations

import json
import os
import random
from pathlib import Path

import modal


APP_NAME = "mini-sglang-cache-cnn-dailymail"
CACHE_MOUNT = "/mnt/mini-sglang-cache"
HF_HOME = f"{CACHE_MOUNT}/huggingface"
SUBSET_DIR = f"{CACHE_MOUNT}/datasets/cnn_dailymail-3.0.0-test-100-seed-0"
MANIFEST_PATH = f"{SUBSET_DIR}/benchmark_manifest.json"
DATASET_REPO = "abisee/cnn_dailymail"
DATASET_REVISION = "96df5e686bee6baa90b8bee7c28b81fa3fa6223d"

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
        "3.0.0",
        split="test",
        revision=DATASET_REVISION,
        cache_dir=f"{HF_HOME}/datasets",
    )
    indices = random.Random(0).sample(range(len(dataset)), 100)
    expected_ids = [dataset[index]["id"] for index in indices]

    subset_path = Path(SUBSET_DIR)
    state_path = subset_path / "state.json"
    if state_path.exists():
        subset = load_from_disk(SUBSET_DIR)
        actual_ids = subset["id"]
        if len(subset) != 100 or actual_ids != expected_ids:
            raise RuntimeError(
                f"Existing frozen subset does not match seed-0 manifest: {SUBSET_DIR}"
            )
    else:
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        subset = dataset.select(indices)
        subset.save_to_disk(SUBSET_DIR)

    manifest = {
        "dataset": DATASET_REPO,
        "revision": DATASET_REVISION,
        "config": "3.0.0",
        "source_split": "test",
        "source_rows": len(dataset),
        "selection": {"method": "random.sample", "seed": 0, "rows": 100},
        "indices": indices,
        "ids": expected_ids,
        "columns": subset.column_names,
        "hf_cache": f"{HF_HOME}/datasets",
        "saved_subset": SUBSET_DIR,
    }
    Path(MANIFEST_PATH).write_text(json.dumps(manifest, indent=2) + "\n")
    cache_volume.commit()

    verified = load_from_disk(SUBSET_DIR)
    if len(verified) != 100 or verified["id"] != expected_ids:
        raise RuntimeError("CNN/DailyMail subset verification failed after Volume commit")

    return json.dumps(
        {
            "status": "ok",
            "source_rows": len(dataset),
            "cached_rows": len(verified),
            "columns": verified.column_names,
            "first_id": verified[0]["id"],
            "last_id": verified[-1]["id"],
            "subset_path": SUBSET_DIR,
            "manifest_path": MANIFEST_PATH,
        },
        indent=2,
    )


@app.local_entrypoint()
def main() -> None:
    print(cache_dataset.remote())
