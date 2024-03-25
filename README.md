# GTCRN
This repository is the official implementation of the ICASSP2024 paper: [GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources](https://ieeexplore.ieee.org/document/10448310). 

Audio examples are available at [Audio examples of GTCRN](https://htmlpreview.github.io/?https://github.com/Xiaobin-Rong/gtcrn_demo/blob/main/index.html).

## About GTCRN
Grouped Temporal Convolutional Recurrent Network (GTCRN) is a speech enhancement model requiring ultralow computational resources, featuring only **23.7 K** parameters and **33.0 MMACs** per second.
Experimental results show that our proposed model not only surpasses RNNoise, a typical lightweight model with similar computational burden, 
but also achieves competitive performance when compared to recent baseline models with significantly higher computational resources requirements.

Note:
* Although the complexity mentioned in the paper is **39.6 MMACs** per second, we find that it can be further reduced to **33.0 MMACs** per second. This reduction can be achieved by modifying only the ERB module, specifically by substituting the invariant mapping from linear bands to ERB bands in the low-frequency dimension, from matrix multiplication to simple concatenation.
* The explicit feature rearrangement layer in the grouped RNN, which is implemented by feature shuffle, can result in an unstreamable model. Therefore, we discard it and implicitly achieve feature rearrangement through the following FC layer in the DPGRNN.


## Pre-trained Models
Pre-trained models are provided in `checkpoints` folder, which were trained on DNS3 and VCTK-DEMAND datasets, respectively.

The inference procedure is presented in `infer.py`.

## Streaming Inference
A streaming GTCRN is provided in `stream` folder, which demonstrates an impressive real-time factor (RTF) of **0.07** on the 12th Gen Intel(R) Core(TM) i5-12400 CPU @ 2.50 GHz.

## Related Repositories
[SEtrain](https://github.com/Xiaobin-Rong/SEtrain): A training code template for DNN-based speech enhancement.

[TRT-SE](https://github.com/Xiaobin-Rong/TRT-SE): An example of how to convert a speech enhancement model into a streaming format and deploy it using ONNX or TensorRT.
