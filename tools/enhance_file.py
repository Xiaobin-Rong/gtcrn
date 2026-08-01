#!/usr/bin/env python3
"""Enhance one WAV file with the official GTCRN PyTorch model."""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gtcrn import GTCRN  # noqa: E402

SAMPLE_RATE = 16_000
N_FFT = 512
HOP_LENGTH = 256
WIN_LENGTH = 512
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "model_trained_on_dns3.tar"


@dataclass
class EnhancementMetrics:
    input_path: str
    output_path: str
    input_sr: int
    input_channels: int
    input_duration: float
    output_sr: int
    output_duration: float
    input_peak: float
    output_peak: float
    processing_time: float
    rtf: float
    device: str
    resampled: bool
    downmixed: bool
    output_gain: float


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Use --device cpu or verify the GPU allocation and PyTorch CUDA build."
        )
    if requested not in {"cpu", "cuda"}:
        raise ValueError("--device must be one of: auto, cpu, cuda")
    return torch.device(requested)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"Checkpoint is empty: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(checkpoint).__name__}")
    if "model" not in checkpoint or not isinstance(checkpoint["model"], dict):
        raise KeyError("Checkpoint does not contain a 'model' state_dict")
    return checkpoint


def read_wav(path: Path) -> tuple[np.ndarray, int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Input WAV does not exist: {path}")
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[0] == 0:
        raise ValueError(f"Input WAV is empty: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Input WAV contains NaN or Inf: {path}")
    channels = int(audio.shape[1])
    if channels > 1:
        print(f"Channel conversion: {channels} channels -> mono (channel mean)")
    mono = audio.mean(axis=1, dtype=np.float32)
    return np.asarray(mono, dtype=np.float32), int(sample_rate), channels


def resample_to_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == SAMPLE_RATE:
        print("Sample-rate conversion: not needed (16000 Hz)")
        return audio
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate: {sample_rate}")
    divisor = math.gcd(sample_rate, SAMPLE_RATE)
    up = SAMPLE_RATE // divisor
    down = sample_rate // divisor
    print(
        f"Sample-rate conversion: {sample_rate} Hz -> {SAMPLE_RATE} Hz "
        f"(polyphase resampling, up={up}, down={down})"
    )
    return np.asarray(resample_poly(audio, up, down), dtype=np.float32)


class GTCRNEnhancer:
    """Load GTCRN once and enhance any number of waveforms or files."""

    def __init__(self, checkpoint: Path | str, device: str = "auto") -> None:
        self.checkpoint_path = Path(checkpoint).expanduser().resolve()
        self.device = resolve_device(device)
        self.model = GTCRN().to(self.device).eval()
        checkpoint_data = load_checkpoint(self.checkpoint_path, self.device)
        incompatible = self.model.load_state_dict(checkpoint_data["model"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Unhandled checkpoint mismatch: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        self.window = torch.hann_window(WIN_LENGTH, device=self.device).sqrt()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(self.device)}")
        print(f"Model parameters: {self.parameter_count:,}")

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def enhance_waveform(self, audio: np.ndarray) -> tuple[np.ndarray, float]:
        if audio.ndim != 1:
            raise ValueError(f"Expected mono waveform, got shape {audio.shape}")
        if audio.size == 0:
            raise ValueError("Cannot enhance an empty waveform")
        if not np.isfinite(audio).all():
            raise ValueError("Input waveform contains NaN or Inf")
        original_length = int(audio.size)
        minimum_length = N_FFT // 2 + 1
        if audio.size < minimum_length:
            print(
                f"Short-input padding: {audio.size} -> {minimum_length} samples "
                f"(output will be trimmed back to {original_length})"
            )
            audio = np.pad(audio, (0, minimum_length - audio.size))

        waveform = torch.from_numpy(np.ascontiguousarray(audio)).to(self.device)
        self._synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            spectrum = torch.view_as_real(
                torch.stft(
                    waveform,
                    n_fft=N_FFT,
                    hop_length=HOP_LENGTH,
                    win_length=WIN_LENGTH,
                    window=self.window,
                    return_complex=True,
                )
            )
            enhanced_spectrum = self.model(spectrum.unsqueeze(0))[0]
            enhanced = torch.istft(
                torch.view_as_complex(enhanced_spectrum.contiguous()),
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                win_length=WIN_LENGTH,
                window=self.window,
                length=int(audio.size),
            )
        self._synchronize()
        elapsed = time.perf_counter() - start
        result = enhanced[:original_length].detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError("Enhanced output contains NaN or Inf")
        if not np.any(np.abs(result) > 1e-12):
            raise RuntimeError("Enhanced output is all zero")
        return result, elapsed

    def process_file(
        self, input_path: Path | str, output_path: Path | str
    ) -> EnhancementMetrics:
        input_path = Path(input_path).expanduser().resolve()
        output_path = Path(output_path).expanduser().resolve()
        audio, input_sr, channels = read_wav(input_path)
        input_duration = audio.size / input_sr
        input_peak = float(np.max(np.abs(audio)))
        converted = resample_to_16k(audio, input_sr)
        enhanced, elapsed = self.enhance_waveform(converted)

        gain = 1.0
        raw_peak = float(np.max(np.abs(enhanced)))
        if raw_peak > 0.999:
            gain = 0.999 / raw_peak
            enhanced *= gain
            print(
                f"Clipping prevention: applied gain {gain:.8f} "
                f"because raw peak was {raw_peak:.8f}"
            )
        else:
            print("Clipping prevention: no gain adjustment needed")
        output_peak = float(np.max(np.abs(enhanced)))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, enhanced, SAMPLE_RATE, subtype="PCM_16")
        saved_info = sf.info(output_path)
        if saved_info.samplerate != SAMPLE_RATE or saved_info.channels != 1:
            raise RuntimeError(f"Unexpected saved WAV format: {saved_info}")

        output_duration = enhanced.size / SAMPLE_RATE
        rtf = elapsed / output_duration
        metrics = EnhancementMetrics(
            input_path=str(input_path),
            output_path=str(output_path),
            input_sr=input_sr,
            input_channels=channels,
            input_duration=input_duration,
            output_sr=SAMPLE_RATE,
            output_duration=output_duration,
            input_peak=input_peak,
            output_peak=output_peak,
            processing_time=elapsed,
            rtf=rtf,
            device=str(self.device),
            resampled=input_sr != SAMPLE_RATE,
            downmixed=channels > 1,
            output_gain=gain,
        )
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print(
            f"Duration: {input_duration:.6f} s input -> "
            f"{output_duration:.6f} s output"
        )
        print(f"Peak: {input_peak:.8f} input -> {output_peak:.8f} output")
        print(f"Inference time: {elapsed:.6f} s")
        print(f"RTF: {rtf:.6f} ({'real-time' if rtf < 1.0 else 'not real-time'})")
        return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    enhancer = GTCRNEnhancer(args.checkpoint, args.device)
    print("Metrics:", asdict(enhancer.process_file(args.input, args.output)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
