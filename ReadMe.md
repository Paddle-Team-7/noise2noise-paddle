# Noise2Noise-Paddle

English | [简体中文](./README_cn.md)

This is an unofficial Paddle implementation of [Noise2Noise](https://arxiv.org/abs/1803.04189) (Lehtinen et al. 2018).

## Contents
1. [Introduction](#introduction)
2. [Reproduction Accuracy](#reproduction-accuracy)
3. [Reprod_Log](#reprod-log)
4. [Dataset](#dataset)
5. [Environment](#environment)
6. [Train](#train)
7. [Test](#test)
8. [Code Structure](#code-structure)
9. [Result](result)

## Introduction

**Reference Code：**[noise2noise](https://github.com/joeylitalien/noise2noise-pytorch)

**Paper：**[Noise2Noise: Learning Image Restoration without Clean Data](https://arxiv.org/abs/1803.04189)


## Reproduction Accuracy
In training, set batch size to 64.

| Index | Raw Paper| Reference Code | Reproduction |
| --- | --- | --- | --- |
| PSNR(gaussian-denoised) | 31.61 | 32.16 | 32.35 |
| PSNR(gaussian-clean) | 31.63 | 32.21 | 32.19 |

## Reprod Log
Based on 'reprod_log' model, the following documents are produced.
```
log_reprod
├── forward_paddle.npy
├── forward_torch.npy
├── metric_paddle.npy
├── metric_torch.npy
├── loss_paddle.npy
├── loss_torch.npy
├── bp_align_paddle.npy
├── bp_align_torch.npy
├── train_align_paddle.npy
├── train_align_torch.npy
```

Based on 'ReprodDiffHelper' model, the following five log files are produced.

```
├── forward_diff.log
├── metric_diff.log
├── loss_diff.log
├── bp_align_diff.log
├── train_align_diff.log
```

## Dataset
The authors use [ImageNet](http://image-net.org/download), but any dataset will do. [COCO 2017](http://cocodataset.org/#download) has a small validation set (1 GB) which can be nicely split into train/valid for easier training. For instance, to obtain a 4200/800 train/valid split you can do:
```
cd data
mkdir train valid test
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip && cd val2017
mv `ls | head -4200` ../train
mv `ls | head -800` ../valid
```

You can also download the full datasets (7 GB) that more or less match the paper, if you have the bandwidth:

```
cd data
mkdir train valid test
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip -j test2017.zip -d train
unzip -j val2017.zip -d valid
```

Add your favorite images to the `data/test` folder. Only a handful will do to visually inspect the denoiser performance.


## Environment
- Frameworks: 
* [PaddlePaddle](https://paddlepaddle.org.cn/) (2.1.2)
* [NumPy](http://www.numpy.org/) (1.14.2)
* [Matplotlib](https://matplotlib.org/) (2.2.3)
* [Pillow](https://pillow.readthedocs.io/en/latest/index.html) (5.2.0)


## Train

See `python3 train.py --h` for list of optional arguments, or `examples/train.sh` for an example.

By default, the model train with noisy targets. To train with clean targets, use `--clean-targets`. To train and validate on smaller datasets, use the `--train-size` and `--valid-size` options. To plot stats as the model trains, use `--plot-stats`; these are saved alongside checkpoints. By default CUDA is not enabled: use the `--cuda` option if you have a GPU that supports it.

### Gaussian noise
The noise parameter is the maximum standard deviation σ.
```
python3 train.py \
  --train-dir ../data/train --train-size 1000 \
  --valid-dir ../data/valid --valid-size 200 \
  --ckpt-save-path ../ckpts \
  --nb-epochs 10 \
  --batch-size 4 \
  --loss l2 \
  --noise-type gaussian \
  --noise-param 25 \
  --crop-size 64 \
  --plot-stats \
  --cuda
```

## Test
Model checkpoints are automatically saved after every epoch. To test the denoiser, provide `test.py` with a PyTorch model (`.pt` file) via the argument `--load-ckpt` and a test image directory via `--data`. The `--show-output` option specifies the number of noisy/denoised/clean montages to display on screen. To disable this, simply remove `--show-output`.

```
python3 test.py \
  --data ../data/test \
  --load-ckpt ../ckpts/gaussian/n2n.pdparams \
  --noise-type gaussian \
  --noise-param 25 \
  --crop-size 0 \
  --show-output 3 \
  --seed 1 \
  --cuda
```

```
python3 test.py \
  --data ../data/test \
  --load-ckpt ../ckpts/gaussian/n2n-clean.pdparams \
  --noise-type gaussian \
  --noise-param 25 \
  --crop-size 0 \
  --show-output 3 \
  --seed 1 \
  --cuda
```

See `python3 test.py --h` for list of optional arguments, or `examples/test.sh` for an example.


## Code Structure

```
├── ckpts  # pdparams and training logs
├── figures  # output
├── log_reprod
├── src
│   ├── datasets.py
│   ├── noise2noise.py
│   ├── render.py
│   ├── test.py
│   ├── train.py
│   ├── unet.py
│   └── utils.py
├── LICENSE
├── README_cn.md
├── ReadMe.md
└── requirements.txt
```

## Result

Gaussian model was trained for 200 epochs with a train/valid split of 2000/400.

<table align="center">
  <tr align="center">
    <th colspan=9>Gaussian noise (σ = 25)</td>
  </tr>
  <tr align="center">
    <td colspan=2>Noisy input (20.34 dB)</td>
    <td colspan=2>Denoised (32.35 dB)</td>
    <td colspan=2>Clean targets (32.19 dB)</td>
    <td colspan=2>Ground truth</td>
  </tr>
  <tr align="center">
    <td colspan=2><img src="figures/n2n-gaussian/monarch-gaussian-noisy.png"></td>
    <td colspan=2><img src="figures/n2n-gaussian/monarch-gaussian-denoised.png"></td>
    <td colspan=2><img src="figures/n2n-gaussian-clean/monarch-gaussian-denoised.png"></td>
    <td colspan=2><img src="data/test/monarch.png"></td>
  </tr> 
</table>
