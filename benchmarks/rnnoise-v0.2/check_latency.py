"""Check RNNoise v0.2 latency and partial-frame handling on five speech clips."""
import argparse
import ctypes
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags

from evaluate import RNNoise

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--noisy', required=True)
parser.add_argument('--rnnoise-library', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
rn = RNNoise(args.rnnoise_library)
files = sorted(Path(args.noisy).glob('*.wav'))[:5]
if len(files) != 5:
    raise ValueError('Need at least five clips')
results = []
for path in files:
    audio, rate = sf.read(path, dtype='float32')
    if rate != 48000 or audio.ndim != 1:
        raise ValueError('Expected 48 kHz mono audio')
    padded = np.pad(audio, (0, (-len(audio)) % 480 + 960)) * np.float32(32768)
    raw = np.empty_like(padded)
    state = rn.lib.rnnoise_create(None)
    if not state:
        raise RuntimeError('rnnoise_create failed')
    try:
        for offset in range(0, len(padded), 480):
            inp = np.ascontiguousarray(padded[offset:offset + 480])
            out = raw[offset:offset + 480]
            rn.lib.rnnoise_process_frame(
                state, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    finally:
        rn.lib.rnnoise_destroy(state)
    correlation = correlate(raw / np.float32(32768), audio, method='fft')
    lag = int(correlation_lags(len(raw), len(audio))[np.argmax(correlation)])
    aligned = rn(audio)
    if lag != 960 or len(aligned) != len(audio):
        raise AssertionError(f'Unexpected latency/length: {path}, {lag}')
    if not np.array_equal(aligned, raw[960:960 + len(audio)] / np.float32(32768)):
        raise AssertionError('Latency compensation changed samples')
    if not np.any(aligned[-480:] != 0):
        raise AssertionError('Final samples were lost')
    results.append({'filename': path.name, 'raw_output_lag_samples_48k': lag,
                    'input_samples': len(audio), 'output_samples': len(aligned)})
Path(args.output).write_text(json.dumps(results, indent=2) + '\n')
print('Passed: 960-sample delay, preserved lengths and tails on five clips')
