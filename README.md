# GTCRN
This repository is the official implementation of the ICASSP2024 paper: [GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources](https://ieeexplore.ieee.org/document/10448310). 

Audio examples are available at [Audio examples of GTCRN](https://htmlpreview.github.io/?https://github.com/Xiaobin-Rong/gtcrn_demo/blob/main/index.html).

## 🔥 News
- [**2026-1-18**] Added a LADSPA plugin for filtering live audio on Linux using pipewire. Please note that LADSPA support is experimental and currently maintained by community contributor [Bruno Gonçalves](https://github.com/bigbruno).
- [**2025-5-27**] A lightweight hybrid dual-channel SE system adapted for low-SNR conditions is released in [H-GTCRN](https://github.com/Max1Wz/H-GTCRN).
- [**2025-3-13**] A quick inference web is built thanks to [Fangjun Kuang](https://github.com/csukuangfj), see in [web](https://huggingface.co/spaces/k2-fsa/wasm-speech-enhancement-gtcrn).
- [**2025-3-10**] Now GTCRN is supported by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) thanks to [Fangjun Kuang](https://github.com/csukuangfj), see [here](https://github.com/k2-fsa/sherpa-onnx/pull/1977).
- [**2025-3-05**] An improved ultra-lightweight SE model named **UL-UNAS** is proposed, see in [repo](https://github.com/Xiaobin-Rong/ul-unas) and [arxiv](https://arxiv.org/abs/2503.00340).

## About GTCRN
Grouped Temporal Convolutional Recurrent Network (GTCRN) is a speech enhancement model requiring ultralow computational resources, featuring only **48.2 K** parameters and **33.0 MMACs** per second.
The original paper compared GTCRN with the 2018 RNNoise baseline and several larger models.
An updated comparison with the default RNNoise v0.2 release model is reported below.

Note:
* The complexity reported in the paper is **23.7K** parameters and **39.6 MMACs** per second; however, we update these values to **48.2K** parameters and **33.0 MMACs** per second here. This modification is due to the inclusion of the ERB module. When accounting for the parameters of the ERB module (even though they are unlearnable), the parameter count increases to 48.2K. By replacing the invariant mapping from linear bands to ERB bands in the low-frequency dimension with simple concatenation instead of matrix multiplication, the MACs per second are reduced to 33 MMACs.
* The explicit feature rearrangement layer in the grouped RNN, which is implemented by feature shuffle, can result in an unstreamable model. Therefore, we discard it and implicitly achieve feature rearrangement through the following FC layer in the DPGRNN.

## Performance

The original RNNoise (2018) comparison predates [RNNoise v0.2](https://github.com/xiph/rnnoise/releases/tag/v0.2), released April 15, 2024. The tables below report a fresh evaluation of its **release-bundled default model**, the supplied GTCRN checkpoints, and the noisy input on all 824 VCTK-DEMAND test clips and all 600 DNS3 blind-test clips.

GTCRN scores higher on VCTK-DEMAND PESQ, STOI, and SI-SNR, and on DNS3 P.808, background quality, and overall quality. RNNoise v0.2 scores higher on DNS3 speech quality (SIG) than GTCRN; the overall-quality gap is smaller than in the historical comparison. These are objective metric estimates, not listening-test MOS.

**Table 1**: VCTK-DEMAND test set, rerun October 4, 2026 (higher is better).
| | SI-SNR | PESQ-WB | STOI |
|:--:|:--:|:--:|:--:|
| Noisy | 8.45 | 1.97 | 0.921 |
| RNNoise v0.2 (release default) | 13.26 | 2.45 | 0.922 |
| GTCRN (VCTK checkpoint) | **18.80** | **2.85** | **0.941** |

**Table 2**: DNS3 blind test set, same rerun (higher is better).
| | DNSMOS-P.808 | BAK | SIG | OVRL |
|:--:|:--:|:--:|:--:|:--:|
| Noisy | 2.964 | 2.646 | **3.198** | 2.333 |
| RNNoise v0.2 (release default) | 3.306 | 3.650 | 3.082 | 2.675 |
| GTCRN (DNS3 checkpoint) | **3.447** | **3.897** | 2.997 | **2.704** |

See [evaluation protocol, reproduction commands, and per-clip results](benchmarks/rnnoise-v0.2/README.md). RNNoise runs at 48 kHz with its fixed 20 ms output delay compensated; metrics use 16 kHz audio. DNSMOS uses Microsoft's regular P.835 calibration and P.808 model. All rows above share the same evaluation protocol. Complexity was not remeasured: the old RNNoise parameter/MAC counts do not describe v0.2.

<details>
<summary>Historical comparisons from the original paper (RNNoise 2018)</summary>

These published results are retained for reference. They are not RNNoise v0.2 results and should not be mixed with the fresh evaluation above.

**Original paper table**: VCTK-DEMAND test set
|    |Para. (M)|MACs (G/s)|SISNR|PESQ|STOI|
|:--:|:-------:|:--------:|:---:|:--:|:--:|
|Noisy|-|-|8.45|1.97|0.921
|RNNoise (2018)|0.06|0.04|-|2.29|-|
|PercepNet (2020)|8.00|0.80|-|2.73|-|
|DeepFilterNet (2022)|1.80|0.35|16.63|2.81|**0.942**|
|S-DCCRN (2022)|2.34|-|-|2.84|0.940|
|GTCRN (proposed)|**0.05**|**0.03**|**18.83**|**2.87**|0.940|
<br>

**Original paper table**: DNS3 blind test set.
|    |Para. (M)|MACs (G/s)|DNSMOS-P.808|BAK|SIG|OVRL|
|:--:|:-------:|:--------:|:----------:|:-:|:-:|:--:|
|Noisy|-|-|2.96|2.65|**3.20**|2.33|
|RNNoise (2018)|0.06|0.04|3.15|3.45|3.00|2.53|
|S-DCCRN (2022)|2.34|-|3.43|-|-|-|
|GTCRN (proposed)|**0.05**|**0.03**|**3.44**|**3.90**|3.00|**2.70**|

</details>

## Pre-trained Models
Pre-trained models are provided in `checkpoints` folder, which were trained on DNS3 and VCTK-DEMAND datasets, respectively.

The inference procedure is presented in `infer.py`.

## Streaming Inference
A streaming GTCRN is provided in `stream` folder, which demonstrates an impressive real-time factor (RTF) of **0.07** on the 12th Gen Intel(R) Core(TM) i5-12400 CPU @ 2.50 GHz.

## Related Repositories
[SEtrain](https://github.com/Xiaobin-Rong/SEtrain): A training code template for DNN-based speech enhancement.

[TRT-SE](https://github.com/Xiaobin-Rong/TRT-SE): An example of how to convert a speech enhancement model into a streaming format and deploy it using ONNX or TensorRT.
