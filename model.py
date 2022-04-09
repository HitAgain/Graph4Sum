#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: HitAgain

import os
os.environ['TF_KERAS'] = '1'
import logging

import numpy as np

from config import Config as hp
from backend import keras
from backend import backend as K
from backend import layers
from backend import optimizers
from util import add_dims, MultiHeadAttention, maxpool
from graph import GraphConv, GraphMaxPool

from data import Data,data_generator


class Graph4Sum(object):

    def __init__(self):
        # hidden_dim = 128
        self.hidden_dim = hp.hidden_dim
        self.vocab_size = hp.vocab_size
        self.graph_dim = hp.graph_dim

    # timedistribute + lstm may hit some shape error just abandon
    # def build_timedistribute(self):
    #     src = keras.layers.Input(shape=[None, None, ], dtype="int32", name="src_input_layer")
    #     tgt = keras.layers.Input(shape=[None, ], dtype="int32", name="tgt_input_layer")
    #     graph = keras.layers.Input(shape=[None, None, ], dtype="float32", name="graph_input_layer")
    #     # layer
    #     SENT_EMB = keras.layers.Embedding(self.vocab_size,
    #                                       self.hidden_dim,
    #                                       trainable=False,
    #                                       mask_zero=True,
    #                                       name='sent_emb_layer'
    #     )
    #     GCN_State = GraphConv(
    #         units= self.graph_dim,
    #         name='graph_conv_layer_1',
    #     )
    #     GCN_Context = GraphConv(
    #         units= self.graph_dim,
    #         name='graph_conv_layer_2',
    #     )
    #     MAX_pool = keras.layers.Lambda(
    #         maxpool, name = "seq_pool_layer"
    #     )
    #     decoder_lstm = keras.layers.LSTM(
    #         units=self.graph_dim,
    #         return_state=False,
    #         return_sequences=True,
    #         name='decoder_lstm'
    #     )
    #     # for loss compute
    #     out_mask = keras.layers.Lambda(
    #         lambda x: K.cast(
    #             K.greater(
    #                 K.expand_dims(
    #                     x,
    #                     2),
    #                 0),
    #             'float32'))(tgt)
    #     # encoder
    #      # shape = [bc_size, sent_nums, sent_length, emb_size]
    #     src_emb = keras.layers.TimeDistributed(SENT_EMB)(src)
    #      # shape = [bc_size, sent_nums, emb_size]
    #     src_emb = keras.layers.TimeDistributed(keras.layers.LSTM(
    #         self.hidden_dim, return_state=False, return_sequences=False, dropout=0.5))(src_emb)
    #      # shape = [bc_size, sent_nums, emb_size]
    #     src_emb = tf.transpose(src_emb, perm=[1,0,2])
    #     decode_state = GCN_State([src_emb, graph])
    #      # shape = [bc_size, emb_size]
    #     decode_state = MAX_pool(decode_state)
    #      # shape = [bc_size, sent_nums, emb_size]
    #     decode_context = GCN_Context([src_emb, graph])
    #      # shape = [bc_size, emb_size]
    #     decode_context = MAX_pool(decode_context)

    #     # decoder
    #      # shape = [bc_size, sent_length, emb_size]
    #     dec_emb = keras.layers.Dropout(0.2)(SENT_EMB(tgt))
    #      # shape = [bc_size, sent_length, emb_size]
    #     decoder_outputs = decoder_lstm(dec_emb, initial_state=[decode_state, decode_context])
    #      # shape = [bc_size, sent_length, emb_size]
    #     final = keras.layers.Dense(128)(decoder_outputs)
    #      # shape = [bc_size, sent_length, emb_size]
    #     final = keras.layers.LeakyReLU(0.2)(final)
    #      # shape = [bc_size, sent_length, vocab_size]
    #     project = keras.layers.Dense(
    #         units=self.vocab_size,
    #         activation='softmax',
    #         name='final_out')(final)
    #     cross_entropy = K.sparse_categorical_crossentropy(
    #         tgt[:, 1:], project[:, :-1])
    #     cross_entropy = K.sum(
    #         cross_entropy * out_mask[:, 1:, 0]) / K.sum(out_mask[:, 1:, 0])
    #     graph_sum_model = keras.models.Model(
    #         name="graph4sum", inputs=[src, tgt, graph], outputs=[project])
    #     graph_sum_model.summary()
    #     # add loss and compile
    #     graph_sum_model.add_loss(cross_entropy)
    #     graph_sum_model.compile(optimizer = optimizers.Adam(1e-3))
    #     return graph_sum_model

    def build(self):
        src_1 = keras.layers.Input(shape=[None, ], dtype="int32", name="src_input_1")
        src_2 = keras.layers.Input(shape=[None, ], dtype="int32", name="src_input_2")
        src_3 = keras.layers.Input(shape=[None, ], dtype="int32", name="src_input_3")
        tgt = keras.layers.Input(shape=[None, ], dtype="int32", name="tgt_input")
        graph = keras.layers.Input(shape=[3, 3, ], dtype="float32", name="graph_input")
        # layer
        sent_emb = keras.layers.Embedding(self.vocab_size,
                                          self.hidden_dim,
                                          trainable=True,
                                          mask_zero=True,
                                          name='sent_emb_layer')

        encoder_lstm_1 = keras.layers.LSTM(self.hidden_dim,
                                           return_state=True,
                                           return_sequences=True,
                                           dropout=0.2,
                                           name = 'encoder_1_layer')

        encoder_lstm_2 = keras.layers.LSTM(self.hidden_dim,
                                           return_state=True,
                                           return_sequences=True,
                                           dropout=0.2,
                                           name = 'encoder_2_layer')

        encoder_lstm_3 = keras.layers.LSTM(self.hidden_dim,
                                           return_state=True,
                                           return_sequences=True,
                                           dropout=0.2,
                                           name = 'encoder_3_layer')

        GCN_State = GraphConv(
            units= self.graph_dim,
            name='graph_conv_state_layer',
        )
        GCN_Context = GraphConv(
            units= self.graph_dim,
            name='graph_conv_context_layer',
        )

        pool_layer = keras.layers.Lambda(
            maxpool, name = "seq_pool"
        )

        ADD_DIMS = keras.layers.Lambda(
            add_dims, name = "add_dims_layer")


        decoder_lstm = keras.layers.LSTM(
            units=self.graph_dim,
            return_state=False,
            return_sequences=True,
            name='decoder_lstm_layer'
        )
        # mask for loss compute
        out_mask = keras.layers.Lambda(
            lambda x: K.cast(
                K.greater(
                    K.expand_dims(
                        x,
                        2),
                    0),
                'float32'))(tgt)

        src_emb_1 = sent_emb(src_1)
        src_emb_2 = sent_emb(src_2)
        src_emb_3 = sent_emb(src_3)

        _, src_context_1, src_state_1 = encoder_lstm_1(src_emb_1)
        _, src_context_2, src_state_2 = encoder_lstm_2(src_emb_2)
        _, src_context_3, src_state_3 = encoder_lstm_3(src_emb_3)

        src_context_1_add_dims = ADD_DIMS(src_context_1)
        src_context_2_add_dims = ADD_DIMS(src_context_2)
        src_context_3_add_dims = ADD_DIMS(src_context_3)

        src_state_1_add_dims = ADD_DIMS(src_state_1)
        src_state_2_add_dims = ADD_DIMS(src_state_2)
        src_state_3_add_dims = ADD_DIMS(src_state_3)

        src_context_graph_input = keras.layers.Concatenate(axis = -2)([src_context_1_add_dims,
                                                            src_context_2_add_dims,
                                                            src_context_3_add_dims])
        decode_context = GCN_Context([src_context_graph_input, graph])
        decode_context_pool = pool_layer(decode_context)

        src_state_graph_input = keras.layers.Concatenate(axis = -2)([src_state_1_add_dims,
                                                          src_state_2_add_dims,
                                                          src_state_3_add_dims])
        decode_state = GCN_State([src_state_graph_input, graph])
        decode_state_pool = pool_layer(decode_state)

        dec_emb = sent_emb(tgt)
        decoder_outputs = decoder_lstm(dec_emb, initial_state=[decode_state_pool, decode_context_pool])
         # shape = [bc_size, sent_length, vocab_size]
        project = keras.layers.Dense(
            units=self.vocab_size,
            activation='softmax',
            name='final_out')(decoder_outputs)
        cross_entropy = K.sparse_categorical_crossentropy(
            tgt[:, 1:], project[:, :-1])
        cross_entropy = K.sum(
            cross_entropy * out_mask[:, 1:, 0]) / K.sum(out_mask[:, 1:, 0])
        graph_sum_model = keras.models.Model(
            name="graph4sum", inputs=[src_1, src_2, src_3, tgt, graph], outputs=[project])
        graph_sum_model.summary()
        # add loss and compile
        graph_sum_model.add_loss(cross_entropy)
        graph_sum_model.compile(optimizer = optimizers.Adam(1e-3))
        return graph_sum_model


if __name__ == '__main__':
    ## check model correction
    Graph4Sum = Graph4Sum()
    logging.debug("build model start")
    model = Graph4Sum.build()
    logging.debug("build model end")
    data = Data("./data/train.txt", "/home/QuerySim/chinese_L-12_H-768_A-12/vocab.txt")
    logging.debug("train data preparing start")
    train_data = data.load_data()
    logging.debug("data example:{}".format(train_data[:3]))
    train_generator = data_generator(train_data, 16)
    logging.debug("train data preparing end")
    logging.info("start training")
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=3,
        verbose=1
    )
    logging.info("finish train")
