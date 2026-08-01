#!/usr/bin/env python3
"""Recursively enhance a directory of WAV files with one GTCRN model load."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import soundfile as sf

from enhance_file import DEFAULT_CHECKPOINT, GTCRNEnhancer

CSV_FIELDS = (
    "input_path",
    "output_path",
    "input_sr",
    "duration",
    "processing_time",
    "rtf",
    "status",
    "error",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV path (default: OUTPUT_DIR/enhance_results.csv)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    if input_dir == output_dir:
        raise ValueError("Input and output directories must be different")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = (
        args.csv.expanduser().resolve()
        if args.csv is not None
        else output_dir / "enhance_results.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path = output_dir / "failed_files.txt"

    wavs = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    )
    print(f"Found {len(wavs)} WAV file(s) under {input_dir}")
    enhancer = GTCRNEnhancer(args.checkpoint, args.device)
    counts = {"success": 0, "skipped": 0, "failed": 0}
    failed_files: list[str] = []
    rows: list[dict[str, object]] = []

    for index, input_path in enumerate(wavs, start=1):
        relative = input_path.relative_to(input_dir)
        output_path = output_dir / relative
        print(f"\n[{index}/{len(wavs)}] {relative}")
        row: dict[str, object] = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "input_sr": "",
            "duration": "",
            "processing_time": "",
            "rtf": "",
            "status": "",
            "error": "",
        }
        try:
            info = sf.info(input_path)
            row["input_sr"] = info.samplerate
            row["duration"] = info.frames / info.samplerate
            if output_path.exists() and not args.overwrite:
                row["status"] = "skipped"
                counts["skipped"] += 1
                print(f"Skipped existing output: {output_path}")
            else:
                metrics = enhancer.process_file(input_path, output_path)
                row["input_sr"] = metrics.input_sr
                row["duration"] = f"{metrics.input_duration:.9f}"
                row["processing_time"] = f"{metrics.processing_time:.9f}"
                row["rtf"] = f"{metrics.rtf:.9f}"
                row["status"] = "success"
                counts["success"] += 1
        except Exception as error:  # Per-file isolation is intentional.
            message = f"{type(error).__name__}: {error}"
            row["status"] = "failed"
            row["error"] = message
            counts["failed"] += 1
            failed_files.append(f"{input_path}\t{message}")
            print(f"FAILED: {message}", file=sys.stderr)
        rows.append(row)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with failed_path.open("w", encoding="utf-8") as handle:
        if failed_files:
            handle.write("\n".join(failed_files) + "\n")
    print(
        "\nSummary: "
        f"success={counts['success']}, skipped={counts['skipped']}, "
        f"failed={counts['failed']}"
    )
    print(f"CSV: {csv_path}")
    print(f"Failed files: {failed_path}")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
