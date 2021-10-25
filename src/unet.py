#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import paddle
import paddle.nn as nn


class UNet(nn.Layer):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net."""

        super(UNet, self).__init__()

        #init weight & bias
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.KaimingNormal())
        bias_attr = paddle.ParamAttr(
            initializer=nn.initializer.Constant(value=0.0))

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2D(in_channels, 48, 3, stride=1, padding=1, weight_attr=weight_attr, bias_attr=bias_attr),
            nn.ReLU(),
            nn.Conv2D(48, 48, 3, padding=1, weight_attr=weight_attr, bias_attr=bias_attr),
            nn.ReLU(),
            nn.MaxPool2D(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2D(48, 48, 3, stride=1, padding=1, weight_attr=weight_attr, bias_attr=bias_attr),
            nn.ReLU(),
            nn.MaxPool2D(2))

        # Layers: enc_conv6, upsample5
        self._block3 = nn.Sequential(
            nn.Conv2D(48, 48, 3, stride=1, padding=1, weight_attr=weight_attr, bias_attr=bias_attr),
            nn.ReLU(),
            nn.Conv2DTranspose(48, 48, 3, stride=2, padding=1, output_padding=1, weight_attr=weight_attr, bias_attr=bias_attr))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.Sequential(
            nn.Conv2D(96, 96, 3, stride=1, padding=1, weight_attr=weight_attr, bias_attr=bias_attr),
            nn.ReLU(),
            nn.Conv2D(96, 96, 3, stride=1, padding=1, weight_attr=weight_attr, bias_attr=bias_attr),
            nn.ReLU(),
            nn.Conv2DTranspose(96, 96, 3, stride=2, padding=1, output_padding=1, weight_attr=weight_attr, bias_attr=bias_attr))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.Sequential(
            nn.Conv2D(144, 96, 3, stride=1, padding=1, weight_attr=weight_attr, bias_attr=bias_attr),
            nn.ReLU(),
            nn.Conv2D(96, 96, 3, stride=1, padding=1, weight_attr=weight_attr, bias_attr=bias_attr),
            nn.ReLU(),
            nn.Conv2DTranspose(96, 96, 3, stride=2, padding=1, output_padding=1, weight_attr=weight_attr, bias_attr=bias_attr))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv2D(96 + in_channels, 64, 3, stride=1, padding=1, weight_attr=weight_attr, bias_attr=bias_attr),
            nn.ReLU(),
            nn.Conv2D(64, 32, 3, stride=1, padding=1, weight_attr=weight_attr, bias_attr=bias_attr),
            nn.ReLU(),
            nn.Conv2D(32, out_channels, 3, stride=1, padding=1, weight_attr=weight_attr, bias_attr=bias_attr),
            nn.LeakyReLU(0.1))

        # Initialize weights
        #self._init_weights()


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.Conv2DTranspose) or isinstance(m, nn.Conv2D):
                m.weight_attr.data = paddle.ParamAttr(
                    name="weight",
                    initializer=nn.initializer.KaimingNormal())
                m.bias_attr.data = paddle.ParamAttr(
                    name="bias",
                    initializer=nn.initializer.Constant(value=0.0))



    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """

        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2(pool2)
        pool4 = self._block2(pool3)
        pool5 = self._block2(pool4)

        # Decoder
        upsample5 = self._block3(pool5)
        concat5 = paddle.concat([upsample5, pool4], axis=1)
        upsample4 = self._block4(concat5)
        concat4 = paddle.concat([upsample4, pool3], axis=1)
        upsample3 = self._block5(concat4)
        concat3 = paddle.concat([upsample3, pool2], axis=1)
        upsample2 = self._block5(concat3)
        concat2 = paddle.concat([upsample2, pool1], axis=1)
        upsample1 = self._block5(concat2)
        concat1 = paddle.concat([upsample1, x], axis=1)

        # Final activation
        return self._block6(concat1)
