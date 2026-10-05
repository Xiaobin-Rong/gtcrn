# RNNoise v0.2 comparison

These are fresh evaluations of the **default model bundled in the official
RNNoise 0.2 release tarball**, GTCRN's supplied checkpoints, and the noisy input.
They replace the 2018 RNNoise baseline in the main README comparison. The original
paper's tables remain explicitly historical; these runs do not amend the paper.

RNNoise [v0.2](https://github.com/xiph/rnnoise/releases/tag/v0.2) was released on
April 15, 2024, after the original comparison. The old VCTK-DEMAND PESQ of 2.29
also appears as **original RNNoise** in the
[2020 PercepNet paper](https://www.isca-archive.org/interspeech_2020/valin20_interspeech.pdf).
The repository does not pin the exact RNNoise revision used for its historical
DNS3 result; its table identifies it as RNNoise (2018), not v0.2.

## Protocol

- Evaluated October 4, 2026, on macOS arm64, CPU, Python 3.12.2; exact Python
  package versions are in `requirements.txt`. GTCRN source base:
  `502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73`.
- VCTK-DEMAND: all **824** paired, 48 kHz mono clips from the official
  [Edinburgh VoiceBank test archives](https://doi.org/10.7488/ds/2117).
  Wideband PESQ at 16 kHz (`pesq`, mode `wb`), standard STOI (not ESTOI), and
  zero-mean SI-SNR; arithmetic mean of per-utterance scores.
- DNS3: all **600** clips in
  `V2_V3_DNSChallenge_Blindset/noisy_blind_testset_v3_challenge_withSNR_16k`
  from Microsoft's [official combined archive download script](https://github.com/microsoft/DNS-Challenge/blob/591184a9fcb2cbdec02520fed81a32bbbf9d73ff/download_dns_v2_v3_blindset.sh).
  Excludes DNS2, extra emotional clips, and mouse-click clips. Regular,
  non-personalized [Microsoft DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/591184a9fcb2cbdec02520fed81a32bbbf9d73ff/DNSMOS)
  `ComputeScore`, at revision `591184a9fcb2cbdec02520fed81a32bbbf9d73ff`:
  `model_v8.onnx` for P.808; `sig_bak_ovr.onnx` and its regular polynomial
  calibration for P.835. Default 9.01-second windows, one-second hops, short-clip
  repetition, per-clip window means followed by an equal-weight clip mean.
- RNNoise processes at 48 kHz with `rnnoise_create(NULL)` (fresh state per clip)
  and float PCM scaled by 32768. No custom/retrained/little model, VAD gating,
  extra gain, or integer PCM round trip. Final partial frames are zero-padded,
  two zero frames drain the pipeline, and the **960-sample / 20 ms** output delay
  is removed before restoring the original sample count. `check_latency.py`
  independently checks raw-output cross-correlation and tail preservation on
  five clips, including partial frames.
- All resampling uses SciPy `resample_poly` with defaults: 48→16 kHz for scoring
  and GTCRN; DNS3 additionally uses 16→48 kHz before RNNoise. No normalization or
  extra postfilter. Outputs are not clipped. DNSMOS receives float32 WAV data.
- GTCRN uses the appropriate `model_trained_on_vctk.tar` or
  `model_trained_on_dns3.tar`, CPU inference, the supplied model, 512-point STFT,
  256 hop, square-root Hann window, and centered STFT. Complex tensor conversions
  replace deprecated PyTorch `return_complex=False` calls; inverse STFT explicitly
  retains input length.
- Every file must score successfully; no skips. The script checks counts, mono
  audio, sample rates, pair lengths, finite scores, and output lengths. CSVs
  retain individual scores and input/clean hashes; JSON summaries include
  checkpoint, scoring-model, script, binary, and CSV hashes.

These are objective metric estimates, not listening-test MOS. Historical numbers
may use different resampling, DNSMOS versions, or preprocessing. Only the fresh
rows were evaluated with this identical protocol. Complexity and speed were not
benchmarked; the old RNNoise 0.06 M / 0.04 G/s figures must not be assigned to the
larger v0.2 model.

The release tarball's bundled weights differ from the external model fetched by
`autogen.sh` in the Git tag checkout. Use the tarball and its hash below to reproduce
**these release-default results**. Current `main` and alternative models are not
covered by this comparison.

## Reproduce

Run from the repository root. Build tools: C compiler, make, patch, curl, unzip.
Download the archives below into `benchmark-work/` and extract both VoiceBank ZIPs
there. Extract only the DNS3 directory listed above from the combined ZIP. The
~1 GB downloads and audio stay outside the committed results.

```sh
mkdir -p benchmark-work
cd benchmark-work
curl -fL https://github.com/xiph/rnnoise/releases/download/v0.2/rnnoise-0.2.tar.gz -o rnnoise-0.2.tar.gz
curl -fL https://github.com/xiph/rnnoise/commit/372f7b4b76cde4ca1ec4605353dd17898a99de38.patch -o build-fix.patch
# Check SHA-256 against the table below before extracting/building.
shasum -a 256 rnnoise-0.2.tar.gz build-fix.patch
tar -xzf rnnoise-0.2.tar.gz
cd rnnoise-0.2
patch -p1 < ../build-fix.patch
./configure --disable-examples
make -j4
cd ..
git clone https://github.com/microsoft/DNS-Challenge.git
git -C DNS-Challenge checkout 591184a9fcb2cbdec02520fed81a32bbbf9d73ff
curl -fL https://datashare.ed.ac.uk/bitstreams/dec213d3-bf57-4777-9663-c24bdce92d5e/download -o clean_testset_wav.zip
curl -fL https://datashare.ed.ac.uk/bitstreams/13c1bfbf-14a6-41db-9b41-8f7310f01ad5/download -o noisy_testset_wav.zip
curl -fL https://dnschallengepublic.blob.core.windows.net/dns3archive/V2_V3_Challenge_Combined_Blindset.zip -o dns_blind.zip
shasum -a 256 *.zip
unzip -q clean_testset_wav.zip
unzip -q noisy_testset_wav.zip
unzip -q dns_blind.zip '*noisy_blind_testset_v3_challenge_withSNR_16k/*'
cd ..
python3.12 -m venv benchmark-work/venv
benchmark-work/venv/bin/pip install -r benchmarks/rnnoise-v0.2/requirements.txt
# On Linux substitute .so for .dylib below.
benchmark-work/venv/bin/python benchmarks/rnnoise-v0.2/check_latency.py \
  --noisy benchmark-work/noisy_testset_wav \
  --rnnoise-library benchmark-work/rnnoise-0.2/.libs/librnnoise.dylib \
  --output benchmark-work/latency-check.json
benchmark-work/venv/bin/python benchmarks/rnnoise-v0.2/evaluate.py \
  --dataset vctk --noisy benchmark-work/noisy_testset_wav \
  --clean benchmark-work/clean_testset_wav \
  --rnnoise-library benchmark-work/rnnoise-0.2/.libs/librnnoise.dylib \
  --output benchmark-work/results
benchmark-work/venv/bin/python benchmarks/rnnoise-v0.2/evaluate.py \
  --dataset dns3 \
  --noisy benchmark-work/V2_V3_DNSChallenge_Blindset/noisy_blind_testset_v3_challenge_withSNR_16k \
  --rnnoise-library benchmark-work/rnnoise-0.2/.libs/librnnoise.dylib \
  --dns-repo benchmark-work/DNS-Challenge --output benchmark-work/results
```

The applied patch is RNNoise's immediate post-release **compilation-only** fix
(`372f7b4b76cde4ca1ec4605353dd17898a99de38`), needed for missing headers/macros
on arm64. It does not change the network weights or denoising algorithm.

| Download / model artifact | SHA-256 |
| --- | --- |
| `rnnoise-0.2.tar.gz` | `90fce4b00b9ff24c08dbfe31b82ffd43bae383d85c5535676d28b0a2b11c0d37` |
| `build-fix.patch` | `c491dfba7784ba027f7293259652053bb63bc834aae693269e4b5cf1dda54b05` |
| Release `src/rnnoise_data.c` | `dece58eabca6a722a0bd9dbc8dfadba27abac1271760fad65e1071f54717dc78` |
| Release `src/rnnoise_data.h` | `09ff880bddd0fc74a2ae0e5ec6c8d65714031b08d0c3f672493acd9e189c5855` |
| `clean_testset_wav.zip` | `b60bfc2b9df38466d8eb45057c087c6e5dee196c67a0ed7bca628ae2a8611c4e` |
| `noisy_testset_wav.zip` | `ba4a893f45b627f15209ba30ab584dd495b3680cd6f216dd08d143a257611993` |
| `dns_blind.zip` | `9c38502bdb243f7c356bf60a5157c184809916c06031629d027cfe76f47fd5b8` |

See [`results/`](results/) for unrounded aggregate and per-clip results. Binary
hashes are specific to this build/platform; model and dataset hashes identify the
portable inputs. Small numerical differences on other platforms are expected.
