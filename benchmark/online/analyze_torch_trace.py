from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def _stats(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean_us": sum(values) / len(values) if values else 0.0,
        "p50_us": _percentile(values, 0.50),
        "p90_us": _percentile(values, 0.90),
        "p99_us": _percentile(values, 0.99),
        "max_us": max(values, default=0.0),
    }


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def analyze(trace_path: Path) -> dict[str, Any]:
    trace = json.loads(trace_path.read_text())
    events = trace.get("traceEvents", [])
    forwards = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") == "user_annotation"
        and str(event.get("name", "")).startswith("minisgl_forward_")
    ]
    if not forwards:
        raise RuntimeError("trace contains no minisgl_forward_* annotations")
    window_start = min(event["ts"] for event in forwards)
    window_end = max(event["ts"] + event["dur"] for event in forwards)

    gpu_categories = {"kernel", "gpu_memcpy", "gpu_memset"}
    gpu_events = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") in gpu_categories
        and event.get("ts", 0) < window_end
        and event.get("ts", 0) + event.get("dur", 0) > window_start
    ]
    intervals = [
        (
            max(window_start, float(event["ts"])),
            min(window_end, float(event["ts"] + event["dur"])),
        )
        for event in gpu_events
    ]
    merged = _merge_intervals(intervals)
    gaps = [merged[index + 1][0] - merged[index][1] for index in range(len(merged) - 1)]
    busy_us = sum(end - start for start, end in merged)
    window_us = window_end - window_start

    forward_durations: dict[str, list[float]] = defaultdict(list)
    forward_batch_sizes: Counter[str] = Counter()
    for event in forwards:
        phase = str(event["name"]).split("_", 3)[2]
        forward_durations[phase].append(float(event["dur"]))
        forward_batch_sizes[str(event["name"])] += 1

    graph_launches = [
        float(event["dur"])
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") == "cuda_runtime"
        and event.get("name") == "cudaGraphLaunch"
        and window_start <= event.get("ts", 0) < window_end
    ]
    return {
        "trace": str(trace_path),
        "profile_window_us": window_us,
        "forward_steps": len(forwards),
        "forward_cpu_duration": {
            phase: _stats(durations) for phase, durations in sorted(forward_durations.items())
        },
        "forward_shapes": dict(forward_batch_sizes.most_common()),
        "cuda_graph_launch_cpu_duration": _stats(graph_launches),
        "gpu_events": len(gpu_events),
        "gpu_busy_us": busy_us,
        "gpu_idle_us": window_us - busy_us,
        "gpu_busy_percent": 100 * busy_us / window_us,
        "inter_gpu_work_gaps": _stats(gaps),
        "gaps_over_10us": sum(gap > 10 for gap in gaps),
        "gaps_over_50us": sum(gap > 50 for gap in gaps),
        "gaps_over_100us": sum(gap > 100 for gap in gaps),
        "largest_gaps_us": sorted(gaps, reverse=True)[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.trace)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
