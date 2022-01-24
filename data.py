#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: HitAgain

from ltp import LTP
import numpy as np

from tokenizer import Tokenizer

class Data(object):
	"""数据处理类"""
	def __init__(self, file_path, vocab_path, batch_size):
		# 语法分析工具
		self.ltp_parse_ = LTP()
		# 原始数据路径
		self.file_path_ = file_path
		# bert字典工具类
		self.tokenizer_ = Tokenizer(vocab_path, do_lower_case=True)
		# batch size
		self.batch_size = batch_size

	def load_data(self):
	    src1, src2, src3, summary = list(), list(), list(), list()
	    with open(self.file_path_, 'r', encoding='utf-8') as f_in:
	        for line in f_in:
	            splits = line.strip('\n').split('>>>')
	            if len(splits) != 4:
	            	print("find one error data just skip")
	            else:
	            	src1.append(splits[1])
	            	src2.append(splits[2])
	            	src3.append(splits[3])
	            	summary.append(summary)
	    assert len(src1) == len(src2) == len(src3) == len(summary)
	    return src1, src2, src3, summary

    def data_generator(self):
        # 文本1/文本2/文本3/摘要
	    a, b, c, d = self.load_data()
	    x, y, z, s = [], [], [], []
	    while True:
	        for a_, b_, c_, d_ in zip(a, b, c, d):
	            x.append(a_)
	            y.append(b_)
	            z.append(c_)
	            s.append(d_)
	            if len(x) == self.batch_size:
	                x_ = batch_encode(x)
	                y_ = batch_encode(y)
	                z_ = batch_encode(z)
	                s_ = batch_encode(s)
	                yield [x_, y_, z_, s_], None
	                x, y, z, s = [], [], [], []

	def batch_encode(self, txt_list):
	    txt_ids_list = []
	    for txt in txt_list:
	        txt_ids_list.append([1] + self.tokenizer_.encode(txt) + [2])
	    txt_ids_list_padded = padding(txt_ids_list)
	    return np.array(txt_ids_list_padded)

	@ staticmethod
	def padding(x):
	    # padding至batch内的最大长度
	    ml = max([len(i) for i in x])
	    return [i + [0] * (ml-len(i)) for i in x]

	@ staticmethod
	def construct_sentence_graph(sentences):
		pass

