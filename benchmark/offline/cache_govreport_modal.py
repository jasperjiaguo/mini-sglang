from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import modal


APP_NAME = "mini-sglang-cache-govreport"
CACHE_MOUNT = "/mnt/mini-sglang-cache"
HF_HOME = f"{CACHE_MOUNT}/huggingface"
SUBSET_DIR = f"{CACHE_MOUNT}/datasets/govreport-test-100-seed-0"
MANIFEST_PATH = f"{SUBSET_DIR}/benchmark_manifest.json"
DATASET_REPO = "ccdv/govreport-summarization"
DATASET_REVISION = "4e21184e01ae8017e2c036e180fe5e541fef60a0"
EXPECTED_ROWS = 100

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
    indices = random.Random(0).sample(range(len(dataset)), EXPECTED_ROWS)
    expected_subset = dataset.select(indices)
    expected_sha256 = dataset_sha256(expected_subset)

    subset_path = Path(SUBSET_DIR)
    state_path = subset_path / "state.json"
    if state_path.exists():
        subset = load_from_disk(SUBSET_DIR)
        if (
            len(subset) != EXPECTED_ROWS
            or dataset_sha256(subset) != expected_sha256
        ):
            raise RuntimeError(
                f"Existing GovReport subset does not match seed-0 manifest: {SUBSET_DIR}"
            )
    else:
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        expected_subset.save_to_disk(SUBSET_DIR)
        subset = expected_subset

    manifest = {
        "dataset": DATASET_REPO,
        "revision": DATASET_REVISION,
        "source_split": "test",
        "source_rows": len(dataset),
        "selection": {
            "method": "random.sample",
            "seed": 0,
            "rows": EXPECTED_ROWS,
        },
        "indices": indices,
        "columns": subset.column_names,
        "content_sha256": expected_sha256,
        "hf_cache": f"{HF_HOME}/datasets",
        "saved_subset": SUBSET_DIR,
    }
    Path(MANIFEST_PATH).write_text(json.dumps(manifest, indent=2) + "\n")
    cache_volume.commit()

    verified = load_from_disk(SUBSET_DIR)
    if (
        len(verified) != EXPECTED_ROWS
        or dataset_sha256(verified) != expected_sha256
    ):
        raise RuntimeError("GovReport subset verification failed after Volume commit")

    return json.dumps(
        {
            "status": "ok",
            "source_rows": len(dataset),
            "cached_rows": len(verified),
            "columns": verified.column_names,
            "content_sha256": expected_sha256,
            "first_index": indices[0],
            "last_index": indices[-1],
            "subset_path": SUBSET_DIR,
            "manifest_path": MANIFEST_PATH,
        },
        indent=2,
    )


@app.local_entrypoint()
def main() -> None:
    print(cache_dataset.remote())
