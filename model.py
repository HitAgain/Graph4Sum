#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: HitAgain

import numpy as np
import tensorflow as tf

from config import Config as hp
from backend import keras
from backend import backend as K
from backend import layers
from util import add_dims, MultiHeadAttention, maxpool
from graph import GraphConv, GraphMaxPool


class Graph4Sum(object):

    def __init__(self):
        # hidden_dim = 128
        self.hidden_dim = hp.hidden_dim
        self.vocab_size = hp.vocab_size
        self.graph_dim = hp.graph_dim

    def build(self):
        src = keras.layers.Input(shape=[None,None,], dtype="int32", name="src_input_layer")
        tgt = keras.layers.Input(shape=[None, ], dtype="int32", name="tgt_input_layer")
        graph = keras.layers.Input(shape=[None, None, ], dtype="float32", name="graph_input_layer")
        # layer
        SENT_EMB = keras.layers.Embedding(self.vocab_size,
                                          self.hidden_dim,
                                          trainable=False,
                                          mask_zero=True,
                                          name='sent_emb_layer'
        )
        GCN_State = GraphConv(
            units= self.graph_dim,
            name='graph_conv_layer_1',
        )
        GCN_Context = GraphConv(
            units= self.graph_dim,
            name='graph_conv_layer_2',
        )
        MAX_pool = keras.layers.Lambda(
            maxpool, name = "seq_pool_layer"
        )
        decoder_lstm = keras.layers.LSTM(
            units=self.graph_dim,
            return_state=False,
            return_sequences=True,
            name='decoder_lstm'
        )
        # for loss compute
        out_mask = keras.layers.Lambda(
            lambda x: K.cast(
                K.greater(
                    K.expand_dims(
                        x,
                        2),
                    0),
                'float32'))(tgt)
        # encoder
        src_emb = keras.layers.TimeDistributed(SENT_EMB)(src)
        src_emb = keras.layers.TimeDistributed(keras.layers.LSTM(
            self.hidden_dim, return_state=False, return_sequences=False, dropout=0.5))(src_emb)
        decode_state = GCN_State([src_emb, graph])
        decode_state = MAX_pool(decode_state)
        decode_context = GCN_Context([src_emb, graph])
        decode_context = MAX_pool(decode_context)
        # decoder
        dec_emb = keras.layers.Dropout(0.2)(SENT_EMB(tgt))
        decoder_outputs = decoder_lstm(dec_emb, initial_state=[decode_state, decode_context])
        final = keras.layers.Dense(128)(decoder_outputs)
        final = keras.layers.LeakyReLU(0.2)(final)
        project = keras.layers.Dense(
            units=self.vocab_size,
            activation='softmax',
            name='final_out')(final)
        cross_entropy = K.sparse_categorical_crossentropy(
            tgt[:, 1:], project[:, :-1])
        cross_entropy = K.sum(
            cross_entropy * out_mask[:, 1:, 0]) / K.sum(out_mask[:, 1:, 0])
        graph_sum_model = keras.models.Model(
            name="graph4sum", inputs=[src, tgt, graph], outputs=[project])
        graph_sum_model.add_loss(cross_entropy)
        graph_sum_model.summary()
        return graph_sum_model

if __name__ == '__main__':
    Graph4Sum = Graph4Sum()
    Graph4Sum.build()
