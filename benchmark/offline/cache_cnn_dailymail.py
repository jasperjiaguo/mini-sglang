from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

DATASET_REPO = "abisee/cnn_dailymail"
DATASET_CONFIG = "3.0.0"
DATASET_SPLIT = "test"
DATASET_REVISION = "96df5e686bee6baa90b8bee7c28b81fa3fa6223d"


def _ids_digest(ids: list[str]) -> str:
    digest = hashlib.sha256()
    for article_id in ids:
        digest.update(article_id.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "datasets" / "cnn_dailymail-3.0.0-test",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path.home() / ".cache" / "huggingface",
    )
    args = parser.parse_args()

    os.environ["HF_HOME"] = str(args.hf_home)
    from datasets import load_dataset, load_from_disk

    if (args.output_dir / "state.json").exists():
        dataset = load_from_disk(str(args.output_dir))
    else:
        dataset = load_dataset(
            DATASET_REPO,
            DATASET_CONFIG,
            split=DATASET_SPLIT,
            revision=DATASET_REVISION,
            cache_dir=str(args.hf_home / "datasets"),
        )
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(args.output_dir))

    if len(dataset) != 11_490:
        raise RuntimeError(f"expected 11,490 test rows, found {len(dataset)}")
    ids = list(dataset["id"])
    manifest = {
        "dataset": DATASET_REPO,
        "revision": DATASET_REVISION,
        "config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "rows": len(dataset),
        "columns": dataset.column_names,
        "first_id": ids[0],
        "last_id": ids[-1],
        "ids_sha256": _ids_digest(ids),
        "saved_dataset": str(args.output_dir),
    }
    (args.output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    verified = load_from_disk(str(args.output_dir))
    if len(verified) != len(dataset) or _ids_digest(list(verified["id"])) != manifest["ids_sha256"]:
        raise RuntimeError("saved CNN/DailyMail dataset verification failed")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
