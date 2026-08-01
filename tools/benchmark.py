#!/usr/bin/env python3
"""Benchmark GTCRN PyTorch CPU/CUDA and streaming ONNX CPU inference."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

from enhance_file import DEFAULT_CHECKPOINT, GTCRNEnhancer, read_wav, resample_to_16k
from stream_onnx import DEFAULT_MODEL, ONNXStreamEnhancer

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "test_wavs" / "mix.wav"
DEFAULT_OUTPUT = REPO_ROOT / "benchmark_results.json"


def summarize(
    name: str,
    times: list[float],
    duration: float,
    *,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "name": name,
        "runs": len(times),
        "audio_duration_seconds": duration,
        "average_seconds": statistics.fmean(times),
        "median_seconds": statistics.median(times),
        "p95_seconds": float(np.percentile(times, 95)),
        "average_rtf": statistics.fmean(times) / duration,
        "median_rtf": statistics.median(times) / duration,
        "p95_rtf": float(np.percentile(times, 95)) / duration,
        "realtime_by_average": statistics.fmean(times) / duration < 1.0,
    }
    if extra:
        result.update(extra)
    return result


def run_timed(
    operation: Callable[[], object], warmup: int, runs: int
) -> list[float]:
    for _ in range(warmup):
        operation()
    times: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        operation()
        times.append(time.perf_counter() - start)
    return times


def prepare_audio(path: Path, target_duration: float) -> np.ndarray:
    audio, sample_rate, channels = read_wav(path)
    if channels > 1:
        print(f"Benchmark input downmixed from {channels} channels")
    audio = resample_to_16k(audio, sample_rate)
    target_samples = max(1, int(round(target_duration * 16_000)))
    repeats = math.ceil(target_samples / audio.size)
    prepared = np.tile(audio, repeats)[:target_samples].astype(np.float32)
    print(
        f"Benchmark audio: {audio.size / 16000:.3f} s sample repeated/truncated "
        f"to {prepared.size / 16000:.3f} s"
    )
    return prepared


def benchmark_torch(
    audio: np.ndarray,
    checkpoint: Path,
    device: str,
    cpu_threads: int,
    warmup: int,
    runs: int,
) -> dict[str, object]:
    import torch

    if device == "cpu":
        torch.set_num_threads(cpu_threads)
    enhancer = GTCRNEnhancer(checkpoint, device)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(enhancer.device)

    def operation() -> None:
        enhancer.enhance_waveform(audio)

    times = run_timed(operation, warmup, runs)
    peak_memory = None
    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(enhancer.device)
    return summarize(
        f"pytorch_{device}",
        times,
        audio.size / 16_000,
        extra={
            "parameter_count": enhancer.parameter_count,
            "peak_gpu_memory_bytes": peak_memory,
            "cpu_threads": cpu_threads if device == "cpu" else None,
            "timing_scope": "STFT + model + ISTFT",
        },
    )


def benchmark_onnx(
    audio: np.ndarray, model: Path, warmup: int, runs: int
) -> dict[str, object]:
    enhancer = ONNXStreamEnhancer(model)
    model_times: list[float] = []

    def operation() -> None:
        _, model_time, _, _ = enhancer.enhance_waveform(audio)
        model_times.append(model_time)

    total_times = run_timed(operation, warmup, runs)
    measured_model_times = model_times[-runs:]
    duration = audio.size / 16_000
    result = summarize(
        "onnxruntime_cpu_stream",
        total_times,
        duration,
        extra={"timing_scope": "STFT + per-frame session.run + ISTFT"},
    )
    result["model_only"] = summarize(
        "onnxruntime_cpu_stream_model_only",
        measured_model_times,
        duration,
        extra={"timing_scope": "sum of per-frame session.run calls"},
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--onnx-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--devices",
        default="cpu,cuda,onnx",
        help="Comma-separated subset of cpu,cuda,onnx",
    )
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    if args.warmup < 5:
        raise ValueError("--warmup must be at least 5")
    if args.runs < 20:
        raise ValueError("--runs must be at least 20")
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    devices = [x.strip().lower() for x in args.devices.split(",") if x.strip()]
    unknown = set(devices) - {"cpu", "cuda", "onnx"}
    if unknown:
        raise ValueError(f"Unknown benchmark devices: {sorted(unknown)}")
    audio = prepare_audio(args.input.expanduser().resolve(), args.duration)
    results: dict[str, object] = {
        "input": str(args.input.expanduser().resolve()),
        "duration_seconds": audio.size / 16_000,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "results": {},
        "errors": {},
    }
    result_items = results["results"]
    error_items = results["errors"]
    assert isinstance(result_items, dict)
    assert isinstance(error_items, dict)
    for device in devices:
        try:
            print(f"\nBenchmarking {device}...")
            if device in {"cpu", "cuda"}:
                result_items[device] = benchmark_torch(
                    audio,
                    args.checkpoint.expanduser().resolve(),
                    device,
                    args.cpu_threads,
                    args.warmup,
                    args.runs,
                )
            else:
                result_items[device] = benchmark_onnx(
                    audio,
                    args.onnx_model.expanduser().resolve(),
                    args.warmup,
                    args.runs,
                )
            print(json.dumps(result_items[device], indent=2))
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            error_items[device] = message
            print(f"{device} benchmark failed: {message}", file=sys.stderr)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nBenchmark JSON: {output_path}")
    return 1 if error_items else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
