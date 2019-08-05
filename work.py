#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : work.py
# Create date : 2019-08-02 20:37
# Modified date : 2019-08-03 22:01
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import time
from collections import Counter

from utils import save_model
from utils import load_model
from utils import flatten_lists
from models.hmm import HMM
from models.crf import CRFModel
from models.bilstm_model import BILSTM_Model
from evaluating import Metrics

HMM_MODEL_PATH = './ckpts/hmm.pkl'
CRF_MODEL_PATH = './ckpts/crf.pkl'
BiLSTM_MODEL_PATH = './ckpts/bilstm.pkl'
BiLSTMCRF_MODEL_PATH = './ckpts/bilstm_crf.pkl'

def _print_metrics(tag_lists, pred):
    REMOVE_O = False  # 在评估的时候是否去除O标记
    metrics = Metrics(tag_lists, pred, remove_O=REMOVE_O)
    metrics.report_scores()  # 打印每个标记的精确度、召回率、f1分数
    metrics.report_confusion_matrix()  # 打印混淆矩阵

def hmm_train(train_data, word2id, tag2id):
    train_word_lists, train_tag_lists = train_data
    hmm_model = HMM(len(tag2id), len(word2id))
    hmm_model.train(train_word_lists, train_tag_lists, word2id, tag2id)
    save_model(hmm_model, HMM_MODEL_PATH)

def hmm_eval(eval_data, word2id, tag2id):
    hmm_model = load_model(HMM_MODEL_PATH)
    word_lists, tag_lists = eval_data
    pred = hmm_model.test(word_lists, word2id, tag2id)
    _print_metrics(tag_lists, pred)
    return pred

def hmm_dev(eval_data, word2id, tag2id):
    return hmm_eval(eval_data, word2id, tag2id)

def crf_train(train_data):
    train_word_lists, train_tag_lists = train_data
    crf_model = CRFModel()
    crf_model.train(train_word_lists, train_tag_lists)
    save_model(crf_model, CRF_MODEL_PATH)

def crf_eval(test_data):
    REMOVE_O = False  # 在评估的时候是否去除O标记
    word_lists, tag_lists = test_data
    crf_model = load_model(CRF_MODEL_PATH)
    pred = crf_model.test(word_lists)
    _print_metrics(tag_lists, pred)
    return pred

def crf_dev(test_data):
    return crf_eval(test_data)

def bilstm_train(train_data, dev_data, word2id, tag2id, crf=False):
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    bilstm_model = BILSTM_Model(vocab_size, out_size, crf=crf)
    bilstm_model.train(train_word_lists, train_tag_lists, dev_word_lists, dev_tag_lists, word2id, tag2id)

    if crf:
        save_model(bilstm_model, BiLSTMCRF_MODEL_PATH)
    else:
        save_model(bilstm_model, BiLSTM_MODEL_PATH)
    print("训练完毕,共用时{}秒.".format(int(time.time()-start)))

def bilstm_eval(test_data, word2id, tag2id, crf=False):
    test_word_lists, test_tag_lists = test_data

    if crf:
        bilstm_model = load_model(BiLSTMCRF_MODEL_PATH)
    else:
        bilstm_model = load_model(BiLSTM_MODEL_PATH)

    pred, tag_lists = bilstm_model.test(test_word_lists, test_tag_lists, word2id, tag2id)
    _print_metrics(tag_lists, pred)
    return pred

def bilstm_crf_train(train_data, dev_data, word2id, tag2id):
    return bilstm_train(train_data, dev_data, word2id, tag2id, crf=True)

def bilstm_crf_eval(test_data, word2id, tag2id):
    return bilstm_eval(test_data, word2id, tag2id, crf=True)

def ensemble_evaluate(results, targets, remove_O=False):
    """ensemble多个模型"""
    for i in range(len(results)):
        results[i] = flatten_lists(results[i])

    pred = []
    for result in zip(*results):
        ensemble_tag = Counter(result).most_common(1)[0][0]
        pred.append(ensemble_tag)

    tag_lists = flatten_lists(targets)
    assert len(pred) == len(tag_lists)

    print("Ensemble 四个模型的结果如下：")
    _print_metrics(tag_lists, pred)
