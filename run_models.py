#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : run_models.py
# Create date : 2019-08-03 17:35
# Modified date : 2019-08-03 17:35
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

from dataset import build_corpus
from dataset import build_vocab

from work import hmm_train
from work import hmm_eval
from work import hmm_dev

from work import crf_train
from work import crf_eval
from work import crf_dev

from utils import extend_maps
from utils import prepocess_data_for_lstmcrf

from work import bilstm_train
from work import bilstm_eval

from work import bilstm_crf_train
from work import bilstm_crf_eval

from work import ensemble_evaluate

def run_hmm(train=True):
    train_word_lists, train_tag_lists = build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev")
    test_word_lists, test_tag_lists = build_corpus("test")
    word2id, tag2id = build_vocab(train_word_lists, train_tag_lists)

    if train:
        print("正在训练HMM模型...")
        hmm_train((train_word_lists, train_tag_lists), word2id, tag2id)
        print("正在评估HMM模型...")
        hmm_dev((dev_word_lists, dev_tag_lists), word2id, tag2id)

    print("正在测试HMM模型...")
    pred = hmm_eval((test_word_lists, test_tag_lists), word2id, tag2id)
    return pred

def run_crf(train=True):
    train_word_lists, train_tag_lists = build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev")
    test_word_lists, test_tag_lists = build_corpus("test")
    word2id, tag2id = build_vocab(train_word_lists, train_tag_lists)
    if train:
        print("正在训练CRF模型...")
        crf_train((train_word_lists, train_tag_lists))
        print("正在评估CRF模型...")
        crf_dev((dev_word_lists, dev_tag_lists))
    print("正在测试CRF模型...")
    pred = crf_eval((test_word_lists, test_tag_lists))
    return pred

def run_lstm(train=True):
    train_word_lists, train_tag_lists = build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev")
    test_word_lists, test_tag_lists = build_corpus("test")
    word2id, tag2id = build_vocab(train_word_lists, train_tag_lists)
    # LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)

    if train:
        print("正在训练双向LSTM模型...")
        bilstm_train((train_word_lists, train_tag_lists), (dev_word_lists, dev_tag_lists), bilstm_word2id, bilstm_tag2id)
    print("正在评估双向LSTM模型...")
    pred = bilstm_eval((test_word_lists, test_tag_lists), bilstm_word2id, bilstm_tag2id)
    return pred

def run_lstm_crf(train=True):
    train_word_lists, train_tag_lists = build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev")
    test_word_lists, test_tag_lists = build_corpus("test")
    word2id, tag2id = build_vocab(train_word_lists, train_tag_lists)

    # 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    # 还需要额外的一些数据处理
    train_word_lists, train_tag_lists = prepocess_data_for_lstmcrf(train_word_lists, train_tag_lists)
    dev_word_lists, dev_tag_lists = prepocess_data_for_lstmcrf(dev_word_lists, dev_tag_lists)
    test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(test_word_lists, test_tag_lists, test=True)

    if train:
        print("正在训练Bi-LSTM+CRF模型...")
        bilstm_crf_train((train_word_lists, train_tag_lists), (dev_word_lists, dev_tag_lists), crf_word2id, crf_tag2id)
    print("正在评估Bi-LSTM+CRF模型...")
    pred = bilstm_crf_eval((test_word_lists, test_tag_lists), crf_word2id, crf_tag2id)
    return pred

def ensemble_pred(hmm_pred, crf_pred, lstm_pred, lstmcrf_pred):
    test_word_lists, test_tag_lists = build_corpus("test")
    ensemble_evaluate([hmm_pred, crf_pred, lstm_pred, lstmcrf_pred], test_tag_lists)

def run(train=True):
    hmm_pred = run_hmm(train)
    crf_pred = run_crf(train)
    lstm_pred = run_lstm(train)
    lstmcrf_pred = run_lstm_crf(train)

    ensemble_pred(hmm_pred, crf_pred, lstm_pred, lstmcrf_pred)
