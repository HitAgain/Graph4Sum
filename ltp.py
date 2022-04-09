#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: HitAgain

import os
import logging

import sys
import codecs
import json
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

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

        try:
            self.srl = SementicRoleLabeller()
            self.srl.load(os.path.join(self.model_dir, "pisrl.model"))
            logging.info("load srl model success")
        except Exception as e:
            logging.error("load pisrl model failed")


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
        self.srl.release()

# pyltp测试函数
def test():
    ltp = LtpParser("/home/ltp_model_3.4.0")
    print("ltp load success")
    words, postags, arcs = ltp.forward('外交部回应世卫组织发布新冠病毒溯源报告')
    print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
    roles = ltp.srl.label(words, postags, arcs)  # 语义角色标注
    res = []
    word_ls = list(words)
    for i in range(len(word_ls)):
        res.append("{}:{}".format(i, word_ls[i]))
    print(res)
    # 打印结果
    for role in roles:
        print("{}:".format(role.index), "".join(["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))

if __name__ == '__main__':
    test()
