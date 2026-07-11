from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import time
import urllib.request
from io import StringIO
from pathlib import Path
from typing import Any

import modal

APP = modal.App("mini-sglang-qwen-cnn-performance-matrix")
REMOTE_ROOT = "/root/mini-sglang"
SOURCE_ROOT = "/workspace/mini-sglang"
_SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = _SCRIPT_PATH.parents[2] if len(_SCRIPT_PATH.parents) > 2 else Path(SOURCE_ROOT)
STABLE_IMAGE_NAME = "mini-sglang-benchmark-cu128-py312:v1"
CACHE_ROOT = "/mnt/mini-sglang-cache"
DATASET_PATH = f"{CACHE_ROOT}/datasets/cnn_dailymail-3.0.0-test-100-seed-0"
CONCURRENCIES = (8, 16, 24, 32, 40, 48, 64)
MODES = (
    {"name": "spec_off", "ngram_size": None, "num_draft_tokens": None},
    {"name": "n3_k2", "ngram_size": 3, "num_draft_tokens": 2},
    {"name": "n3_k3", "ngram_size": 3, "num_draft_tokens": 3},
)

CACHE = modal.Volume.from_name("mini-sglang-cache", environment_name="worktrials")

IMAGE = (
    modal.Image.from_name(
        STABLE_IMAGE_NAME,
        environment_name="worktrials",
    )
    .add_local_dir(
        str(REPO_ROOT),
        SOURCE_ROOT,
        copy=True,
        ignore=[".git/**", ".venv/**", ".hf-cache/**", "**/__pycache__/**"],
    )
)


def _cache_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": f"{CACHE_ROOT}/huggingface",
            "XDG_CACHE_HOME": CACHE_ROOT,
            "FLASHINFER_WORKSPACE_BASE": f"{CACHE_ROOT}/flashinfer",
            "TVM_FFI_CACHE_DIR": f"{CACHE_ROOT}/tvm-ffi",
            "TORCH_EXTENSIONS_DIR": f"{CACHE_ROOT}/torch_extensions",
            "TRITON_CACHE_DIR": f"{CACHE_ROOT}/triton",
            "MINISGL_DISABLE_OVERLAP_SCHEDULING": "1",
            "PATH": f"{REMOTE_ROOT}/.venv/bin:" + env["PATH"],
            "PYTHONPATH": f"{SOURCE_ROOT}/python:{SOURCE_ROOT}",
        }
    )
    return env


def _wait_for_server(port: int, process: subprocess.Popen[str], timeout: float = 600) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"server did not become ready at {url}")


def _server_command(model: str, port: int, mode: dict[str, Any]) -> list[str]:
    command = [
        f"{REMOTE_ROOT}/.venv/bin/python",
        "-m",
        "minisgl",
        "--model-path",
        model,
        "--attention-backend",
        "fi",
        "--cache-type",
        "radix",
        "--page-size",
        "1",
        "--cuda-graph-max-bs",
        "0",
        "--max-running-requests",
        "64",
        "--max-seq-len-override",
        "1056",
        "--max-prefill-length",
        "49152",
        "--port",
        str(port),
    ]
    if mode["ngram_size"] is not None:
        command.extend(
            [
                "--spec-decoding",
                "ngram",
                "--spec-decoding-config",
                json.dumps(
                    {
                        "ngram_size": mode["ngram_size"],
                        "num_draft_tokens": mode["num_draft_tokens"],
                    },
                    separators=(",", ":"),
                ),
            ]
        )
    return command


def _benchmark_command(port: int, concurrency: int, output_dir: Path) -> list[str]:
    return [
        f"{REMOTE_ROOT}/.venv/bin/python",
        f"{SOURCE_ROOT}/benchmark/online/bench_qwen.py",
        "--workload",
        "cnn",
        "--port",
        str(port),
        "--dataset-path",
        DATASET_PATH,
        "--output-dir",
        str(output_dir),
        "--num-requests",
        str(concurrency),
        "--warmup-requests",
        "8",
        "--warmup-max-tokens",
        "32",
        "--max-input-tokens",
        "768",
        "--max-tokens",
        "256",
        "--no-ignore-eos",
        "--seed",
        "0",
        "--no-progress",
    ]


def _csv_text(rows: list[dict[str, Any]]) -> str:
    output = StringIO()
    fields = (
        "mode",
        "concurrency",
        "request_tpot_mean_ms",
        "concurrency_normalized_decode_rate_tokens_per_second",
        "observed_decode_throughput_tokens_per_second",
        "ttft_mean_ms",
        "output_token_throughput",
    )
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _svg_plot(rows: list[dict[str, Any]]) -> str:
    width, height = 1000, 700
    left, right, top, bottom = 105, 40, 60, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    xs = [row["request_tpot_mean_ms"] for row in rows]
    ys = [row["concurrency_normalized_decode_rate_tokens_per_second"] for row in rows]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = max((x_max - x_min) * 0.08, 0.1)
    y_pad = max((y_max - y_min) * 0.08, 1.0)
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = max(0.0, y_min - y_pad), y_max + y_pad

    def px(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def py(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    colors = {"spec_off": "#4c78a8", "n3_k2": "#f58518", "n3_k3": "#54a24b"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#222}.axis{stroke:#333;stroke-width:1.5}'
        '.grid{stroke:#ddd;stroke-width:1}.series{fill:none;stroke-width:2.5}.point{stroke:white;'
        'stroke-width:1.5}</style>',
        '<text x="500" y="30" text-anchor="middle" font-size="20">'
        'Qwen3-8B CNN decode throughput vs request TPOT</text>',
    ]
    for index in range(6):
        x_value = x_min + (x_max - x_min) * index / 5
        x_pos = px(x_value)
        parts.extend(
            [
                f'<line class="grid" x1="{x_pos:.1f}" y1="{top}" x2="{x_pos:.1f}" '
                f'y2="{height-bottom}"/>',
                f'<text x="{x_pos:.1f}" y="{height-bottom+25}" text-anchor="middle" '
                f'font-size="12">{x_value:.2f}</text>',
            ]
        )
        y_value = y_min + (y_max - y_min) * index / 5
        y_pos = py(y_value)
        parts.extend(
            [
                f'<line class="grid" x1="{left}" y1="{y_pos:.1f}" x2="{width-right}" '
                f'y2="{y_pos:.1f}"/>',
                f'<text x="{left-12}" y="{y_pos+4:.1f}" text-anchor="end" '
                f'font-size="12">{y_value:.0f}</text>',
            ]
        )
    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" '
            f'y2="{height-bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
            f'<text x="{left+plot_width/2:.1f}" y="{height-28}" text-anchor="middle" '
            'font-size="15">Mean request TPOT (ms/token)</text>',
            f'<text x="25" y="{top+plot_height/2:.1f}" text-anchor="middle" font-size="15" '
            'transform="rotate(-90 25 '
            f'{top+plot_height/2:.1f})">Concurrency-normalized decode rate (token/s)</text>',
        ]
    )
    for legend_index, mode in enumerate(MODES):
        name = mode["name"]
        color = colors[name]
        mode_rows = sorted(
            (row for row in rows if row["mode"] == name), key=lambda row: row["concurrency"]
        )
        points = " ".join(
            f'{px(row["request_tpot_mean_ms"]):.1f},{py(row["concurrency_normalized_decode_rate_tokens_per_second"]):.1f}'
            for row in mode_rows
        )
        parts.append(f'<polyline class="series" stroke="{color}" points="{points}"/>')
        for row in mode_rows:
            x_pos = px(row["request_tpot_mean_ms"])
            y_pos = py(row["concurrency_normalized_decode_rate_tokens_per_second"])
            parts.extend(
                [
                    f'<circle class="point" cx="{x_pos:.1f}" cy="{y_pos:.1f}" r="5" '
                    f'fill="{color}"/>',
                    f'<text x="{x_pos+7:.1f}" y="{y_pos-7:.1f}" font-size="11">'
                    f'bs={row["concurrency"]}</text>',
                ]
            )
        legend_x = left + legend_index * 150
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="{height-8}" x2="{legend_x+25}" y2="{height-8}" '
                f'stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x+32}" y="{height-3}" font-size="13">{name}</text>',
            ]
        )
    parts.append("</svg>")
    return "\n".join(parts)


@APP.function(
    image=IMAGE,
    gpu="H100!",
    timeout=4 * 60 * 60,
    volumes={CACHE_ROOT: CACHE},
)
def run_matrix(model: str = "Qwen/Qwen3-8B") -> dict[str, Any]:
    env = _cache_env()
    run_id = time.strftime("qwen3_8b_cnn_matrix_%Y%m%d_%H%M%S")
    root = Path(CACHE_ROOT) / "benchmarks" / run_id
    root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []

    for mode_index, mode in enumerate(MODES):
        mode_dir = root / mode["name"]
        mode_dir.mkdir()
        server_log_path = mode_dir / "server.log"
        port = 19190 + mode_index
        command = _server_command(model, port, mode)
        with server_log_path.open("w") as server_log:
            server_log.write("COMMAND=" + json.dumps(command) + "\n")
            server_log.flush()
            process = subprocess.Popen(
                command,
                cwd=SOURCE_ROOT,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                _wait_for_server(port, process)
                for concurrency in CONCURRENCIES:
                    point_dir = mode_dir / f"bs{concurrency}"
                    point_dir.mkdir()
                    client_log = point_dir / "client.log"
                    completed = subprocess.run(
                        _benchmark_command(port, concurrency, point_dir),
                        cwd=SOURCE_ROOT,
                        env=env,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )
                    client_log.write_text(completed.stdout)
                    if completed.returncode:
                        raise RuntimeError(
                            f"{mode['name']} bs={concurrency} failed with "
                            f"status {completed.returncode}; see {client_log}"
                        )
                    summary = json.loads((point_dir / "summary.json").read_text())
                    rows.append(
                        {
                            "mode": mode["name"],
                            "concurrency": concurrency,
                            "request_tpot_mean_ms": summary["request_tpot_ms"]["mean"],
                            "concurrency_normalized_decode_rate_tokens_per_second": summary[
                                "concurrency_normalized_decode_throughput"
                            ]["tokens_per_second"],
                            "observed_decode_throughput_tokens_per_second": summary[
                                "effective_total_decode_throughput"
                            ]["tokens_per_second"],
                            "ttft_mean_ms": summary["ttft_ms"]["mean"],
                            "output_token_throughput": summary["output_token_throughput"],
                        }
                    )
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()

    matrix = {
        "run_id": run_id,
        "model": model,
        "gpu": "H100",
        "stable_image": STABLE_IMAGE_NAME,
        "workload": "cnn_dailymail_summarization",
        "max_input_tokens": 768,
        "max_output_tokens": 256,
        "ignore_eos": False,
        "attention_backend": "fi",
        "cuda_graph_max_bs": 0,
        "overlap_scheduling": False,
        "concurrencies": list(CONCURRENCIES),
        "modes": list(MODES),
        "rows": rows,
        "volume_output_dir": str(root),
    }
    matrix_json = json.dumps(matrix, indent=2) + "\n"
    csv_text = _csv_text(rows)
    svg_text = _svg_plot(rows)
    (root / "matrix.json").write_text(matrix_json)
    (root / "matrix.csv").write_text(csv_text)
    (root / "decode_throughput_vs_tpot.svg").write_text(svg_text)
    CACHE.commit()
    return {"matrix_json": matrix_json, "csv": csv_text, "svg": svg_text, **matrix}


@APP.local_entrypoint()
def main(
    output_dir: str = "benchmark/result/qwen_cnn_performance_matrix",
    model: str = "Qwen/Qwen3-8B",
) -> None:
    result = run_matrix.remote(model)
    destination = Path(output_dir) / result["run_id"]
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "matrix.json").write_text(result.pop("matrix_json"))
    (destination / "matrix.csv").write_text(result.pop("csv"))
    (destination / "decode_throughput_vs_tpot.svg").write_text(result.pop("svg"))
    print(json.dumps({"local_output_dir": str(destination), **result}, indent=2))
