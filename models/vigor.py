import torch
from torchvision import models
import torch.nn as nn


class SAFA_smi(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        vgg_sar = models.vgg16_bn(pretrained=True)
        vgg_sar.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
        vgg_opt = models.vgg16_bn(pretrained=True)
        vgg_opt.maxpool2 = nn.AdaptiveMaxPool2d((1,1))

        height = 20
        width = 20
        self.weights1 = nn.Parameter(torch.randn(height*width, int(height * width / 2),args.dim))
        self.bias1 = nn.Parameter(size)
        self.weights2 = nn.Parameter(size)


    def forward(self, *input: Any, **kwargs: Any) -> T_co:
        return super().forward(*input, **kwargs)


    def spatial_aware(input_feature, dimension, trainable, name):
        batch, height, width, channel = input_feature.shape
        vec1 = input_feature.max(-1).reshape(-1, height * width)

        weight1 = tf.get_variable(name='weights1', shape=[height * width, int(height * width / 2), dimension],
                                trainable=trainable,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.005),
                                regularizer=tf.contrib.layers.l2_regularizer(0.01))
        bias1 = tf.get_variable(name='biases1', shape=[1, int(height * width / 2), dimension],
                                trainable=trainable, initializer=tf.constant_initializer(0.1),
                                regularizer=tf.contrib.layers.l1_regularizer(0.01))
        # vec2 = tf.matmul(vec1, weight1) + bias1
        vec2 = tf.einsum('bi, ijd -> bjd', vec1, weight1) + bias1

        weight2 = tf.get_variable(name='weights2', shape=[int(height * width / 2), height * width, dimension],
                                trainable=trainable,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.005),
                                regularizer=tf.contrib.layers.l2_regularizer(0.01))
        bias2 = tf.get_variable(name='biases2', shape=[1, height * width, dimension],
                                trainable=trainable, initializer=tf.constant_initializer(0.1),
                                regularizer=tf.contrib.layers.l1_regularizer(0.01))
        vec3 = tf.einsum('bjd, jid -> bid', vec2, weight2) + bias2

        return vec3
