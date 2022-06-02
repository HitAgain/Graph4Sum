#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: HitAgain

import os
import numpy as np
import tensorflow as tf
from graph import GraphConv, GraphMaxPool
from util import SearcherWrapper

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class Encoder(tf.keras.layers.Layer):
    def __init__(self, enc_hidden_dim, pretrain_weights):
        """
        enc_units:encoder hidden dim
        pretrain_weights:pretrain c2v
        """
        super(Encoder, self).__init__()
        self.enc_hidden_dim = enc_hidden_dim
        self.pretrain_weights = pretrain_weights
        if self.pretrain_weights:
            self.vocab_size = np.shape(self.pretrain_weights)[0]
            self.embedding_dim = np.shape(self.pretrain_weights)[1]
            self.embedding = tf.keras.layers.Embedding(self.vocab_size,
                                                       self.embedding_dim,
                                                       weights = [self.pretrain_weights],
                                                       trainable = False,
                                                       mask_zero = True)
        else:
            self.vocab_size = 21128
            self.embedding_dim = 128
            self.embedding = tf.keras.layers.Embedding(self.vocab_size,
                                                       self.embedding_dim,
                                                       trainable = False,
                                                       mask_zero = True)

        self.gru = tf.keras.layers.GRU(self.enc_hidden_dim,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self, inp):
        return tf.zeros((tf.shape(inp)[0], self.enc_hidden_dim))

class Decoder(tf.keras.layers.Layer):
    def __init__(self, dec_hidden_dim, pretrain_weights):
        super(Decoder, self).__init__()
        self.dec_hidden_dim = dec_hidden_dim
        self.pretrain_weights = pretrain_weights
        if self.pretrain_weights:
            self.vocab_size = np.shape(self.pretrain_weights)[0]
            self.embedding_dim = np.shape(self.pretrain_weights)[1]
            self.embedding = tf.keras.layers.Embedding(self.vocab_size,
                                                       self.embedding_dim,
                                                       weights = [self.pretrain_weights],
                                                       trainable = False,
                                                       mask_zero = True)
        else:
            self.vocab_size = 21128
            self.embedding_dim = 128
            self.embedding = tf.keras.layers.Embedding(self.vocab_size,
                                                       self.embedding_dim,
                                                       trainable = False,
                                                       mask_zero = True)

        self.attention = BahdanauAttention(self.dec_hidden_dim)
        self.gru = tf.keras.layers.GRU(self.dec_hidden_dim,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(self.vocab_size)


    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        # shape = [bz, hidden_dim]
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state

class GraphSeq2Seq(tf.keras.Model):
    def __init__(self, enc_hidden_dim, dec_hidden_dim, graph_hidden_dim, pretrain_weights, end_token_idx=3):
        super(GraphSeq2Seq, self).__init__()
        self.end_token_idx = end_token_idx
        self.encoder_1 = Encoder(enc_hidden_dim, pretrain_weights)
        self.encoder_2 = Encoder(enc_hidden_dim, pretrain_weights)
        self.encoder_3 = Encoder(enc_hidden_dim, pretrain_weights)
        self.decoder = Decoder(dec_hidden_dim, pretrain_weights) 
        self.gcn =  GraphConv(units = graph_hidden_dim)

    def call(self, inputs, training=True):
        #inp, tar = inputs
        # 输入
        inp1, inp2, inp3, graph, tar = inputs

        enc_hidden_1 = self.encoder_1.initialize_hidden_state(inp1)
        enc_output_1, enc_hidden_1 = self.encoder_1(inp1, enc_hidden_1)

        enc_hidden_2 = self.encoder_2.initialize_hidden_state(inp2)
        enc_output_2, enc_hidden_2 = self.encoder_1(inp1, enc_hidden_2)

        enc_hidden_3 = self.encoder_3.initialize_hidden_state(inp3)
        enc_output_3, enc_hidden_3 = self.encoder_1(inp1, enc_hidden_3)


        enc_hidden_graph_input = tf.keras.layers.Concatenate(axis = -2)(
                [tf.expand_dims(enc_hidden_1, axis=-2),
                 tf.expand_dims(enc_hidden_2, axis=-2),
                 tf.expand_dims(enc_hidden_3, axis=-2)]
        )
        enc_hidden_graph_output = self.gcn([enc_hidden_graph_input, graph])
        dec_hidden = tf.math.reduce_max(enc_hidden_graph_output, axis = 1)
        enc_output = tf.keras.layers.Concatenate(axis = -2)(
                [enc_output_1,
                 enc_output_2,
                 enc_output_3]
        )
        predict_tokens = list()
        for t in range(0, tar.shape[1]):
            dec_input = tf.dtypes.cast(tf.expand_dims(tar[:, t], 1), tf.float32) 
            predictions, dec_hidden = self.decoder(dec_input, dec_hidden, enc_output)
            predict_tokens.append(tf.dtypes.cast(predictions, tf.float32))
        return tf.stack(predict_tokens, axis=1)

    def call_encoder(self, inp1, inp2, inp3, graph):

        enc_hidden_1 = self.encoder_1.initialize_hidden_state(inp1)
        enc_output_1, enc_hidden_1 = self.encoder_1(inp1, enc_hidden_1)

        enc_hidden_2 = self.encoder_2.initialize_hidden_state(inp2)
        enc_output_2, enc_hidden_2 = self.encoder_1(inp1, enc_hidden_2)

        enc_hidden_3 = self.encoder_3.initialize_hidden_state(inp3)
        enc_output_3, enc_hidden_3 = self.encoder_1(inp1, enc_hidden_3)

        enc_hidden_graph_input = tf.keras.layers.Concatenate(axis = -2)(
                [tf.expand_dims(enc_hidden_1, axis=-2),
                 tf.expand_dims(enc_hidden_2, axis=-2),
                 tf.expand_dims(enc_hidden_3, axis=-2)]
        )
        enc_hidden_graph_output = self.gcn([enc_hidden_graph_input, graph])
        dec_hidden = tf.math.reduce_max(enc_hidden_graph_output, axis = 1)
        enc_output = tf.keras.layers.Concatenate(axis = -2)(
                [enc_output_1,
                 enc_output_2,
                 enc_output_3]
        )
        return enc_output, dec_hidden

    def call_docoder(self, dec_hidden, dec_input, enc_output):
        predictions, dec_hidden = self.decoder.call(dec_input, dec_hidden, enc_output)
        return predictions, dec_hidden, enc_output

if __name__ == '__main__':
    gs2s = GraphSeq2Seq(128, 256, 256, None)
    engine = SearcherWrapper(gs2s, 35, 2, 3, 0, 128, 256)
    tf.saved_model.save(engine, "./Graph4SumServing", signatures = {"GreedyGenerate": engine.greedy_search})
