# Noise2Noise-Paddle

[English](./ReadMe.md) | 简体中文

   * [Nois2Noise-Paddle](#noise2noise-paddle)
      * [一、简介](#一简介)
      * [二、复现精度](#二复现精度)
      * [三、对齐日志](#三对齐日志)
      * [四、数据集](#四数据集)
      * [五、环境依赖](#五环境依赖)
      * [六、快速开始](#六快速开始)
         * [step1: clone](#step1-clone)
         * [step2: 下载数据](#step2-下载数据)
         * [step3: 训练](#step3-训练)
         * [step4: 测试与评估](#step4-测试与评估)
      * [七、代码结构与详细说明](#七代码结构)
      * [八、结果展示](#八结果展示)
      * [九、模型信息](#九模型信息)


## 一、简介

【2021飞桨启航菁英计划】本项目使用Paddle复现 [Noise2Noise](https://arxiv.org/abs/1803.04189) (Lehtinen et al. 2018).

**参考代码：**[noise2noise](https://github.com/joeylitalien/noise2noise-pytorch)

**论文：**[Noise2Noise: Learning Image Restoration without Clean Data](https://arxiv.org/abs/1803.04189)


## 二、复现精度

| 指标 | 原论文| 参考代码 | 复现 |
| --- | --- | --- | --- |
| PSNR(gaussian-denoised) | 31.61 | 32.16 | 32.35 |
| PSNR(gaussian-clean) | 31.63 | 32.21 | 32.19 |


## 三、对齐日志

在论文复现中，基于reprod_log的结果记录模块，产出下面若干文件。

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

基于reprod_log的ReprodDiffHelper模块，产出下面5个日志文件。

```
├── forward_diff.log
├── metric_diff.log
├── loss_diff.log
├── bp_align_diff.log
├── train_align_diff.log
```

## 四、数据集
本项目使用[COCO 2017](http://cocodataset.org/#download)的验证集(1 GB)，并将该数据集划分为本程序的训练集和验证集。数据集划分的操作详见快速开始部分。

将测试图片放到 `data/test` 路径。 本项目是用的是参考的pytorch项目所用的图片。


## 五、环境依赖
- 框架: 
* [PaddlePaddle](https://paddlepaddle.org.cn/) (2.1.2)
* [NumPy](http://www.numpy.org/) (1.14.2)
* [Matplotlib](https://matplotlib.org/) (2.2.3)
* [Pillow](https://pillow.readthedocs.io/en/latest/index.html) (5.2.0)


## 六、快速开始

### step1: clone

```bash
# clone this repo
git clone https://github.com/Paddle-Team-7/noise2noise-paddle
```

### step2: 下载数据

下载COCO验证集数据并划分为4200/8000的训练集/测试集。

```
cd data
mkdir train valid test
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip && cd val2017
mv `ls | head -4200` ../train
mv `ls | head -800` ../valid
```

### step3: 训练

Gaussian噪声

```bash
python3 train.py \
  --train-dir ../data/train --train-size 2000 \
  --valid-dir ../data/valid --valid-size 400 \
  --ckpt-save-path ../ckpts \
  --nb-epochs 100 \
  --batch-size 4 \
  --loss l2 \
  --noise-type gaussian \
  --noise-param 25 \
  --crop-size 64 \
  --plot-stats \
  --cuda
```

Clean targets

```bash
python3 train.py \
  --train-dir ../data/train --train-size 2000 \
  --valid-dir ../data/valid --valid-size 400 \
  --ckpt-save-path ../ckpts \
  --nb-epochs 100 \
  --batch-size 4 \
  --loss l2 \
  --noise-type gaussian \
  --noise-param 25 \
  --crop-size 64 \
  --plot-stats \
  --cuda \
  --clean-targets
```

### step4: 测试与评估

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

## 七、代码结构

```
├── ckpts  # 模型权重和训练日志
├── figures  # 模型输出的图片
├── log_reprod  # 对齐日志
├── src
│   ├── datasets.py  # 数据集文件
│   ├── noise2noise.py
│   ├── render.py
│   ├── test.py  # 测试程序
│   ├── train.py  # 训练程序
│   ├── unet.py  # 网络结构
│   └── utils.py  # 工具类
├── LICENSE
├── README_cn.md
├── ReadMe.md
└── requirements.txt
```

#八、结果展示

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

## 九、模型信息

|  信息   |  说明 |
|  ----  |  ----  |
| 作者 | guguguzi&WangChen0902 |
| 时间 | 2021.10.24 |
| 框架版本 | Paddle 2.1.2 |
| 应用场景 | 图像去噪 |
