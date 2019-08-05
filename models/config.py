#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : config.py
# Create date : 2019-08-03 15:14
# Modified date : 2019-08-03 21:44
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

# 设置lstm训练参数
class TrainingConfig(object):
    batch_size = 64
    # 学习速率
    lr = 0.001
    epoches = 30
    print_step = 5

class LSTMConfig(object):
    emb_size = 128  # 词向量的维数
    hidden_size = 128  # lstm隐向量的维数
