#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  HitAgain
# @Date:  2021/1/29 16:21

from backend import backend as K
from backend import keras
import tensorflow as tf


def add_dims(x):
    y = K.expand_dims(x, axis=-2)
    return y

def maxpool(x):
    return K.max(x, 1)

def sequence_masking(x, mask, mode=0, axis=None):
    """为序列条件mask的函数
    mask: 形如(batch_size, seq_len)的0-1矩阵；
    mode: 如果是0，则直接乘以mask；
          如果是1，则在padding部分减去一个大正数。
    axis: 序列所在轴，默认为1；
    heads: 相当于batch这一维要被重复的次数。
    """
    if mask is None or mode not in [0, 1]:
        return x
    else:
        if axis is None:
            axis = 1
        if axis == -1:
            axis = K.ndim(x) - 1
        assert axis > 0, 'axis muse be greater than 0'
        for _ in range(axis - 1):
            mask = K.expand_dims(mask, 1)
        for _ in range(K.ndim(x) - K.ndim(mask) - axis + 1):
            mask = K.expand_dims(mask, K.ndim(mask))
        if mode == 0:
            return x * mask
        else:
            return x - (1 - mask) * 1e12


class MultiHeadAttention(keras.layers.Layer):
    """多头注意力机制
    """

    def __init__(self,
                 heads,
                 head_size,
                 key_size=None,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        self.out_dim = heads * head_size
        self.key_size = key_size if key_size else head_size
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.supports_masking = True

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.q_dense = keras.layers.Dense(units=self.key_size * self.heads,
                                          kernel_initializer=self.kernel_initializer)
        self.k_dense = keras.layers.Dense(units=self.key_size * self.heads,
                                          kernel_initializer=self.kernel_initializer)
        self.v_dense = keras.layers.Dense(units=self.out_dim,
                                          kernel_initializer=self.kernel_initializer)
        self.o_dense = keras.layers.Dense(units=self.out_dim,
                                          kernel_initializer=self.kernel_initializer)

    def call(self, inputs, mask=None):
        """实现多头注意力
        q_mask: 对输入的query序列的mask。
                主要是将输出结果的padding部分置0。
        v_mask: 对输入的value序列的mask。
                主要是防止attention读取到padding信息。
        a_mask: 对attention矩阵的mask。
                不同的attention mask对应不同的应用。
        """
        q, k, v = inputs[:3]
        # 处理mask
        v_mask = K.cast(mask[0], "float32")
        # 线性变换
        qw = self.q_dense(q)
        kw = self.k_dense(k)
        vw = self.v_dense(v)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(q)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(k)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(v)[1], self.heads, self.head_size))
        # Attention
        a = tf.einsum('bjhd,bkhd->bhjk', qw, kw) / self.key_size**0.5
        a = sequence_masking(a, v_mask, 1, -1)
        a = K.softmax(a)
        # 完成输出
        o = tf.einsum('bhjk,bkhd->bjhd', a, vw)
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.o_dense(o)
        return o
