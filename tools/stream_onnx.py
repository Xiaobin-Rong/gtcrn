#!/usr/bin/env python3
"""Run the official streaming GTCRN ONNX model on CPU."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16_000
N_FFT = 512
HOP_LENGTH = 256
WIN_LENGTH = 512
DEFAULT_MODEL = REPO_ROOT / "stream" / "onnx_models" / "gtcrn_simple.onnx"
DEFAULT_INPUT = REPO_ROOT / "stream" / "test_wavs" / "mix.wav"
DEFAULT_OUTPUT = REPO_ROOT / "stream" / "test_wavs" / "enh_onnx.wav"
CACHE_SHAPES = {
    "conv_cache": (2, 1, 16, 16, 33),
    "tra_cache": (2, 3, 1, 1, 16),
    "inter_cache": (2, 1, 33, 16),
}


@dataclass
class StreamMetrics:
    input_path: str
    output_path: str
    model_path: str
    sample_rate: int
    input_duration: float
    output_duration: float
    frames: int
    frame_length: int
    hop_length: int
    model_time: float
    total_time: float
    model_rtf: float
    total_rtf: float
    output_peak: float
    correlation_with_offline: float | None = None
    mae_with_offline: float | None = None


class ONNXStreamEnhancer:
    def __init__(self, model_path: Path | str = DEFAULT_MODEL) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file() or self.model_path.stat().st_size == 0:
            raise FileNotFoundError(f"ONNX model is missing or empty: {self.model_path}")
        model = onnx.load(self.model_path)
        onnx.checker.check_model(model)
        print(f"ONNX IR version: {model.ir_version}")
        print(f"ONNX opsets: {[(item.domain, item.version) for item in model.opset_import]}")
        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        self.input_meta = self.session.get_inputs()
        self.output_meta = self.session.get_outputs()
        print("Providers:", self.session.get_providers())
        print("Inputs:", [(x.name, x.shape, x.type) for x in self.input_meta])
        print("Outputs:", [(x.name, x.shape, x.type) for x in self.output_meta])
        expected_inputs = {"mix", *CACHE_SHAPES}
        actual_inputs = {item.name for item in self.input_meta}
        if actual_inputs != expected_inputs:
            raise RuntimeError(
                f"Unexpected ONNX inputs: expected {sorted(expected_inputs)}, "
                f"got {sorted(actual_inputs)}"
            )
        self.window = torch.hann_window(WIN_LENGTH).sqrt()

    @staticmethod
    def initial_state() -> dict[str, np.ndarray]:
        return {
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in CACHE_SHAPES.items()
        }

    def enhance_waveform(
        self, audio: np.ndarray
    ) -> tuple[np.ndarray, float, float, int]:
        if audio.ndim != 1 or audio.size == 0:
            raise ValueError(f"Expected a non-empty mono waveform, got {audio.shape}")
        if not np.isfinite(audio).all():
            raise ValueError("Input waveform contains NaN or Inf")
        start_total = time.perf_counter()
        with torch.inference_mode():
            spectrum = torch.view_as_real(
                torch.stft(
                    torch.from_numpy(np.ascontiguousarray(audio)),
                    n_fft=N_FFT,
                    hop_length=HOP_LENGTH,
                    win_length=WIN_LENGTH,
                    window=self.window,
                    return_complex=True,
                )
            ).unsqueeze(0)
        frames = spectrum.numpy()
        state = self.initial_state()
        outputs: list[np.ndarray] = []
        model_time = 0.0
        for frame_index in range(frames.shape[2]):
            feed = {
                "mix": frames[:, :, frame_index : frame_index + 1, :],
                **state,
            }
            start_model = time.perf_counter()
            result = self.session.run(None, feed)
            model_time += time.perf_counter() - start_model
            outputs.append(result[0])
            state = {
                "conv_cache": result[1],
                "tra_cache": result[2],
                "inter_cache": result[3],
            }
        enhanced_spectrum = np.concatenate(outputs, axis=2)[0]
        enhanced_complex = torch.view_as_complex(
            torch.from_numpy(np.ascontiguousarray(enhanced_spectrum))
        )
        with torch.inference_mode():
            enhanced = torch.istft(
                enhanced_complex,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                win_length=WIN_LENGTH,
                window=self.window,
                length=int(audio.size),
            )
        total_time = time.perf_counter() - start_total
        result_audio = enhanced.numpy().astype(np.float32)
        if not np.isfinite(result_audio).all():
            raise RuntimeError("Streaming ONNX output contains NaN or Inf")
        if not np.any(np.abs(result_audio) > 1e-12):
            raise RuntimeError("Streaming ONNX output is all zero")
        return result_audio, model_time, total_time, int(frames.shape[2])


def read_official_input(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Official streaming input must be 16000 Hz, got {sample_rate}: {path}"
        )
    if audio.shape[1] != 1:
        raise ValueError(
            f"Official streaming input must be mono, got {audio.shape[1]} channels"
        )
    return audio[:, 0]


def compare_audio(
    streaming: np.ndarray, offline_path: Path | None
) -> tuple[float | None, float | None]:
    if offline_path is None:
        return None, None
    offline, sample_rate = sf.read(offline_path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Offline comparison WAV is not 16000 Hz: {offline_path}")
    offline_mono = offline.mean(axis=1)
    length = min(streaming.size, offline_mono.size)
    if length == 0:
        raise ValueError("Cannot compare empty audio")
    correlation = float(np.corrcoef(streaming[:length], offline_mono[:length])[0, 1])
    mae = float(np.mean(np.abs(streaming[:length] - offline_mono[:length])))
    return correlation, mae


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--compare-offline", type=Path, default=None)
    parser.add_argument("--metrics-json", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    audio = read_official_input(input_path)
    enhancer = ONNXStreamEnhancer(args.model)
    enhanced, model_time, total_time, frames = enhancer.enhance_waveform(audio)
    gain = 1.0
    peak = float(np.max(np.abs(enhanced)))
    if peak > 0.999:
        gain = 0.999 / peak
        enhanced *= gain
        peak = float(np.max(np.abs(enhanced)))
        print(f"Clipping prevention gain: {gain:.8f}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, enhanced, SAMPLE_RATE, subtype="PCM_16")
    sf.read(output_path, dtype="float32")
    offline_path = (
        args.compare_offline.expanduser().resolve()
        if args.compare_offline is not None
        else None
    )
    correlation, mae = compare_audio(enhanced, offline_path)
    duration = audio.size / SAMPLE_RATE
    metrics = StreamMetrics(
        input_path=str(input_path),
        output_path=str(output_path),
        model_path=str(Path(args.model).expanduser().resolve()),
        sample_rate=SAMPLE_RATE,
        input_duration=duration,
        output_duration=enhanced.size / SAMPLE_RATE,
        frames=frames,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
        model_time=model_time,
        total_time=total_time,
        model_rtf=model_time / duration,
        total_rtf=total_time / duration,
        output_peak=peak,
        correlation_with_offline=correlation,
        mae_with_offline=mae,
    )
    payload = json.dumps(asdict(metrics), indent=2, ensure_ascii=False)
    print(payload)
    if args.metrics_json is not None:
        metrics_path = args.metrics_json.expanduser().resolve()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
