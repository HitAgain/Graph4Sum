#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: HitAgain

import numpy as np

from ltp import LtpParser
from tokenizer import Tokenizer
from snippets import DataGenerator, sequence_padding

STARTTOKEN = 1
ENDTOKEN = 2

class Data(object):

  def __init__(self, file_path, vocab_path):
    # ltp tool
    self.LtpParser = LtpParser("/home/ltp_model_3.4.0")
    # data path
    self.file_path_ = file_path
    # bert_vocab
    self.tokenizer_ = Tokenizer(vocab_path, do_lower_case=True)

    # data load
  def load_data(self):
    samples = []
    with open(self.file_path_, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            splits = line.strip('\n').split('\t')
            if len(splits) != 4:
              print("find one error data just skip")
            else:
              sent1_ids = [STARTTOKEN] + self.tokenizer_.encode(splits[0]) + [ENDTOKEN]
              sent2_ids = [STARTTOKEN] + self.tokenizer_.encode(splits[1]) + [ENDTOKEN]
              sent3_ids = [STARTTOKEN] + self.tokenizer_.encode(splits[2]) + [ENDTOKEN]
              sumry_ids = [STARTTOKEN] + self.tokenizer_.encode(splits[3]) + [ENDTOKEN]
              graph = self.construct_sentence_graph([splits[0], splits[1], splits[2]])
              samples.append((sent1_ids, sent2_ids, sent3_ids, sumry_ids, graph))
    return samples

  def construct_sentence_graph(self, sentences):
    nums = len(sentences)
    graph = [[0] * nums for i in range(nums)]
    seg_words = [list(self.LtpParser.segmentor.segment(sentence)) for sentence in sentences]
    for i in range(nums - 1):
      for j in range(i+1, nums, 1):
        graph[i][j] = len(list(set(seg_words[i]).intersection(set(seg_words[j]))))/max(len(seg_words[i]), len(seg_words[j]))
        graph[j][i] = graph[i][j]
    return graph


class data_generator(DataGenerator):
    def __iter__(self, random=False):
      batch_sent_1, batch_sent_2, batch_sent_3, batch_summary, batch_graph = [], [], [], [], []
      for is_end, (sent_1_id, sent_2_id, sent_3_id, summary_id, graph) in self.sample(random):
          batch_sent_1.append(sent_1_id)
          batch_sent_2.append(sent_2_id)
          batch_sent_3.append(sent_3_id)
          batch_summary.append(summary_id)
          batch_graph.append(graph)
          if len(batch_sent_1) == self.batch_size or is_end:
              batch_sent_1_padded = sequence_padding(batch_sent_1)
              batch_sent_2_padded = sequence_padding(batch_sent_2)
              batch_sent_3_padded = sequence_padding(batch_sent_3)
              batch_summary_padded = sequence_padding(batch_summary)
              batch_graph_np =  np.asarray(batch_graph).astype('float32')
              yield [batch_sent_1_padded, batch_sent_2_padded, batch_sent_3_padded, batch_summary_padded, batch_graph_np], None
              batch_sent_1, batch_sent_2, batch_sent_3, batch_summary, batch_graph = [], [], [], [], []

