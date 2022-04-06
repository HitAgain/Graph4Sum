#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: HitAgain

import os
import logging

from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller

class LtpParser(object):
    def __init__(self, model_dir):
        ## ltp模型保存路径
        self.model_dir = model_dir
        try:
            self.segmentor = Segmentor()
            self.segmentor.load(os.path.join(self.model_dir, "cws.model"))
            logging.info("load cws model success")
        except Exception as e:
            logging.error("load cws model failed")
        try:
            self.postagger = Postagger()
            self.postagger.load(os.path.join(self.model_dir, "pos.model"))
            logging.info("load pos model success")
        except Exception as e:
            logging.error("load pos model failed") 
        try:
            self.parser = Parser()
            self.parser.load(os.path.join(self.model_dir, "parser.model"))
            logging.info("load parser model success")
        except Exception as e:
            logging.error("load parser model failed")


    def forward(self, sentence):
        words = self.segmentor.segment(sentence)
        postags  = self.postagger.postag(words)
        arcs = self.parser.parse(words, postags)
        logging.info("======parser success=========")
        return words, postags, arcs

    def __del__(self):
        self.segmentor.release()
        self.postagger.release()
        self.parser.release()
