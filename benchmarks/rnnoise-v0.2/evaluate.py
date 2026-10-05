"""Reproduce RNNoise v0.2 / GTCRN comparisons; see README.md for protocol."""
import argparse
import csv
import ctypes
import hashlib
import importlib.util
import json
import math
import platform
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gtcrn import GTCRN


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resample(audio, source, target):
    common = math.gcd(source, target)
    return resample_poly(audio, target // common, source // common).astype(np.float32)


class RNNoise:
    def __init__(self, library):
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        self.lib.rnnoise_create.argtypes = [ctypes.c_void_p]
        self.lib.rnnoise_create.restype = ctypes.c_void_p
        self.lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        self.lib.rnnoise_get_frame_size.restype = ctypes.c_int
        self.lib.rnnoise_process_frame.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.rnnoise_process_frame.restype = ctypes.c_float
        if self.lib.rnnoise_get_frame_size() != 480:
            raise ValueError('Expected RNNoise v0.2 480-sample frames')

    def __call__(self, audio):
        # v0.2: one delayed spectrum + one overlap-add frame = 960 samples.
        # Pad the final partial frame, drain two frames, remove fixed latency.
        length = len(audio)
        padded = np.pad(audio, (0, (-length) % 480 + 960)) * np.float32(32768)
        output = np.empty_like(padded)
        state = self.lib.rnnoise_create(None)  # release default model, fresh per clip
        if not state:
            raise RuntimeError('rnnoise_create failed')
        try:
            for offset in range(0, len(padded), 480):
                inp = np.ascontiguousarray(padded[offset:offset + 480])
                out = output[offset:offset + 480]
                self.lib.rnnoise_process_frame(
                    state, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        finally:
            self.lib.rnnoise_destroy(state)
        return output[960:960 + length] / np.float32(32768)


def initialize(config):
    global CONFIG, MODEL, RN, SCORER
    CONFIG = config
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    MODEL = GTCRN().eval()
    checkpoint = ROOT / 'checkpoints' / (
        'model_trained_on_vctk.tar' if config['dataset'] == 'vctk' else 'model_trained_on_dns3.tar')
    MODEL.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'])
    RN = RNNoise(config['rnnoise_library'])
    SCORER = None
    if config['dataset'] == 'dns3':
        script = Path(config['dns_repo']) / 'DNSMOS' / 'dnsmos_local.py'
        spec = importlib.util.spec_from_file_location('dnsmos_local', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Keep Microsoft's scoring methods unchanged, limit ORT worker threads.
        SCORER = module.ComputeScore.__new__(module.ComputeScore)
        options = module.ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        models = script.parent / 'DNSMOS'
        SCORER.onnx_sess = module.ort.InferenceSession(
            str(models / 'sig_bak_ovr.onnx'), options, providers=['CPUExecutionProvider'])
        SCORER.p808_onnx_sess = module.ort.InferenceSession(
            str(models / 'model_v8.onnx'), options, providers=['CPUExecutionProvider'])


def gtcrn(audio):
    window = torch.hann_window(512).pow(0.5)
    with torch.inference_mode():
        spectrum = torch.stft(torch.from_numpy(audio), 512, 256, 512, window, return_complex=True)
        enhanced = MODEL(torch.view_as_real(spectrum)[None])[0].contiguous()
        return torch.istft(torch.view_as_complex(enhanced), 512, 256, 512, window,
                           length=len(audio)).numpy()


def si_snr(reference, estimate):
    reference = reference.astype(np.float64) - np.mean(reference)
    estimate = estimate.astype(np.float64) - np.mean(estimate)
    projected = np.dot(estimate, reference) * reference / np.dot(reference, reference)
    return float(10 * np.log10(np.sum(projected ** 2) / np.sum((estimate - projected) ** 2)))


def evaluate(path):
    path = Path(path)
    noisy, rate = sf.read(path, dtype='float32')
    if noisy.ndim != 1 or rate != (48000 if CONFIG['dataset'] == 'vctk' else 16000):
        raise ValueError(f'Unexpected audio format: {path}, {rate}, {noisy.shape}')
    audio16 = resample(noisy, rate, 16000) if rate != 16000 else noisy
    audio48 = resample(noisy, rate, 48000) if rate != 48000 else noisy
    outputs = {'Noisy': audio16, 'RNNoise v0.2': resample(RN(audio48), 48000, 16000),
               'GTCRN': gtcrn(audio16)}
    rows = []
    if CONFIG['dataset'] == 'vctk':
        from pesq import pesq
        from pystoi import stoi
        clean_path = Path(CONFIG['clean']) / path.name
        clean, clean_rate = sf.read(clean_path, dtype='float32')
        if clean.ndim != 1 or clean_rate != rate or len(clean) != len(noisy):
            raise ValueError(f'Clean/noisy mismatch: {path}')
        reference = resample(clean, rate, 16000)
    for name, enhanced in outputs.items():
        if len(enhanced) != len(audio16) or not np.all(np.isfinite(enhanced)):
            raise ValueError(f'Invalid output: {path}, {name}')
        row = {'filename': path.name, 'model': name, 'samples_16k': len(audio16),
               'input_sha256': sha256(path)}
        if CONFIG['dataset'] == 'vctk':
            row.update(clean_sha256=sha256(clean_path),
                       SISNR=si_snr(reference, enhanced),
                       PESQ=float(pesq(16000, reference, enhanced, 'wb')),
                       STOI=float(stoi(reference, enhanced, 16000, extended=False)))
        else:
            with tempfile.TemporaryDirectory() as temp:
                wav = Path(temp) / 'audio.wav'
                sf.write(wav, enhanced, 16000, subtype='FLOAT')
                score = SCORER(str(wav), 16000, False)
            row.update({k: float(score[k]) for k in ('P808_MOS', 'BAK', 'SIG', 'OVRL')})
        if not all(np.isfinite(v) for v in row.values() if isinstance(v, float)):
            raise ValueError(f'Invalid score: {path}, {name}')
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['vctk', 'dns3'], required=True)
    parser.add_argument('--noisy', required=True)
    parser.add_argument('--clean')
    parser.add_argument('--rnnoise-library', required=True)
    parser.add_argument('--dns-repo')
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--expected-files', type=int)
    args = parser.parse_args()
    if args.dataset == 'vctk' and not args.clean:
        parser.error('--clean is required for vctk')
    if args.dataset == 'dns3' and not args.dns_repo:
        parser.error('--dns-repo is required for dns3')
    files = sorted(Path(args.noisy).glob('*.wav'))
    expected = args.expected_files or (824 if args.dataset == 'vctk' else 600)
    if len(files) != expected:
        raise ValueError(f'Expected {expected} clips, found {len(files)}')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             initargs=(vars(args),)) as pool:
        for index, result in enumerate(pool.map(evaluate, files), 1):
            rows.extend(result)
            if index % 20 == 0 or index == len(files):
                print(f'{args.dataset}: {index}/{len(files)}', flush=True)
    with (output / f'{args.dataset}.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = ('SISNR', 'PESQ', 'STOI') if args.dataset == 'vctk' else ('P808_MOS', 'BAK', 'SIG', 'OVRL')
    summary = {'dataset': args.dataset, 'clips': len(files), 'platform': platform.platform(),
               'python': platform.python_version(), 'models': {}}
    for model in ('Noisy', 'RNNoise v0.2', 'GTCRN'):
        selected = [row for row in rows if row['model'] == model]
        summary['models'][model] = {metric: float(np.mean([row[metric] for row in selected]))
                                    for metric in metrics}
    summary['artifact_sha256'] = {'csv': sha256(output / f'{args.dataset}.csv'),
                                  'evaluate.py': sha256(__file__),
                                  'rnnoise_library': sha256(args.rnnoise_library)}
    checkpoint = ROOT / 'checkpoints' / ('model_trained_on_vctk.tar' if args.dataset == 'vctk'
                                         else 'model_trained_on_dns3.tar')
    summary['artifact_sha256']['gtcrn_checkpoint'] = sha256(checkpoint)
    if args.dataset == 'dns3':
        base = Path(args.dns_repo) / 'DNSMOS'
        for relative in ('dnsmos_local.py', 'DNSMOS/sig_bak_ovr.onnx', 'DNSMOS/model_v8.onnx'):
            summary['artifact_sha256'][relative] = sha256(base / relative)
    (output / f'{args.dataset}.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['models'], indent=2))


if __name__ == '__main__':
    main()
