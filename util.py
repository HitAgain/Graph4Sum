#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  HitAgain
# inference module for serving


import tensorflow as tf

class SearcherWrapper(tf.Module):
    """Provide search functions for seq2seq models"""

    def __init__(self,
                 model: tf.keras.Model,
                 max_sequence_length: int, 
                 start_id: int, 
                 end_id: int, 
                 pad_id: int,
                 enc_hidden_dim: int,
                 dec_hidden_dim: int):
        """
        :param model: seq2seq model instance.
        :param max_sequence_length: max sequence length of decoded sequences.
        :param start_id: bos id for decoding.
        :param end_id: eos id for decoding.
        :param pad_id: when a sequence is shorter thans other sentences, the back token ids of the sequence is filled pad id.
        """
        self.model = model
        self.max_sequence_length = max_sequence_length
        self.start_id = start_id
        self.end_id = end_id
        self.pad_id = pad_id
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim

    ## Greedy Search
    ## @tf.function(input_signature=[tf.TensorSpec([None, None], tf.int32)])
    @tf.function(input_signature = [tf.TensorSpec([None, None], tf.int32, name='inp1'),
                                    tf.TensorSpec([None, None], tf.int32, name='inp2'),
                                    tf.TensorSpec([None, None], tf.int32, name='inp3'),
                                    tf.TensorSpec([None, 3, 3], tf.float32, name='graph')])
    def greedy_search(self,
                      encoder_inp_1: tf.int32,
                      encoder_inp_2: tf.int32,
                      encoder_inp_3: tf.int32,
                      encoder_inp_graph: tf.float32) -> tf.Tensor:
        """
        Generate sentences using decoder by beam searching.
        :param encoder_input: seq2seq model inputs [BatchSize, EncoderSequenceLength].
        :return: generated tensor shaped. and ppl value of each generated sentences
        """
        batch_size = tf.shape(encoder_inp_1)[0]
        # encoder_output and dec_hidden_state precomputed
        enc_output, dec_hidden = self.model.call_encoder(encoder_inp_1, encoder_inp_2, encoder_inp_3, encoder_inp_graph)
        decoder_final_res = tf.fill([batch_size, 1], self.start_id)
        cur_dec_input = tf.fill([batch_size, 1], self.start_id)
        log_perplexity = tf.fill([batch_size, 1], 0.0)
        sequence_lengths = tf.fill([batch_size, 1], self.max_sequence_length)
        is_ended = tf.zeros([batch_size, 1], tf.bool)

        def _cond(enc_output, dec_hidden, decoder_final_res, cur_dec_input, is_ended, log_perplexity, sequence_lengths):
            return tf.shape(decoder_final_res)[1] < self.max_sequence_length and not tf.reduce_all(is_ended)

        def _body(enc_output, dec_hidden, decoder_final_res, cur_dec_input, is_ended, log_perplexity, sequence_lengths):
            # [BatchSize, VocabSize]
            predictions, dec_hidden, enc_output = self.model.call_docoder(dec_hidden, cur_dec_input, enc_output)
            output = predictions
            #output = tf.nn.log_softmax(output, axis=1)
            log_probs, new_tokens = tf.math.top_k(output)
            log_probs, new_tokens = tf.cast(log_probs, log_perplexity.dtype), tf.cast(new_tokens, tf.int32)
            log_perplexity = tf.where(is_ended, log_perplexity, log_perplexity + log_probs)
            new_tokens = tf.where(is_ended, self.pad_id, new_tokens)
            is_ended = tf.logical_or(is_ended, new_tokens == self.end_id)
            sequence_lengths = tf.where(new_tokens == self.end_id, tf.shape(decoder_final_res)[1] + 1, sequence_lengths)
            decoder_final_res = tf.concat((decoder_final_res, new_tokens), axis=1)
            cur_dec_input = new_tokens
            return enc_output, dec_hidden, decoder_final_res, cur_dec_input, is_ended, log_perplexity, sequence_lengths

        enc_output, dec_hidden, decoder_final_res, cur_dec_input, is_ended, log_perplexity, sequence_lengths = tf.while_loop(
            _cond,
            _body,
            [enc_output, dec_hidden, decoder_final_res, cur_dec_input, is_ended, log_perplexity, sequence_lengths],
            shape_invariants=[
                tf.TensorSpec([None, None, self.enc_hidden_dim], tf.float32),
                tf.TensorSpec([None, self.dec_hidden_dim], tf.float32),
                tf.TensorSpec([None, None], tf.int32),
                tf.TensorSpec([None, 1], tf.int32),
                tf.TensorSpec(is_ended.get_shape(), is_ended.dtype),
                tf.TensorSpec(log_perplexity.get_shape(), log_perplexity.dtype),
                tf.TensorSpec(sequence_lengths.get_shape(), sequence_lengths.dtype),
            ],
        )
        return decoder_final_res
