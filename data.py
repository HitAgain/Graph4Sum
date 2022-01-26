#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: HitAgain

from ltp import LTP
import numpy as np

from tokenizer import Tokenizer
from snippets import DataGenerator

STARTTOKEN = 1
ENDTOKEN = 2

class Data(object):

	def __init__(self, file_path, vocab_path, batch_size):
		# ltp
		self.ltp_parse_ = LTP()
		# data path
		self.file_path_ = file_path
		# bert_vocab
		self.tokenizer_ = Tokenizer(vocab_path, do_lower_case=True)
		# batch_size
		self.batch_size = batch_size

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
	              samples.append((sent1_ids, sent2_ids, sent3_ids, sumry_ids))
	    return samples

	@ staticmethod
	def construct_sentence_graph(sentences):
		  pass

class data_generator(DataGenerator):
    def __iter__(self, random=False):
      batch_sent_1, batch_sent_2, batch_sent_3, batch_summary = [], [], [], []
      for is_end, (sent_1_id, sent_2_id, sent_3_id, summary_id) in self.sample(random):
          batch_sent_1.append(sent_1_id)
          batch_sent_2.append(sent_2_id)
          batch_sent_3.append(sent_3_id)
          batch_summary.append(summary_id)
          if len(batch_sent_1) == self.batch_size or is_end:
              batch_sent_1_padded = sequence_padding(batch_sent_1)
              batch_sent_2_padded = sequence_padding(batch_sent_2)
              batch_sent_3_padded = sequence_padding(batch_sent_3)
              batch_summary_padded = sequence_padding(batch_summary)
              yield [batch_sent_1_padded, batch_sent_2_padded, batch_sent_3_padded, batch_summary_padded], None
              batch_sent_1, batch_sent_2, batch_sent_3, batch_summary = [], [], [], []
