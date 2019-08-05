#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : dataset.py
# Create date : 2019-08-02 18:22
# Modified date : 2019-08-02 18:23
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

from os.path import join
from codecs import open

def build_corpus(split, data_dir="./ResumeNER"):
    assert split in ['train', 'dev', 'test']

    word_lists = []
    tag_lists = []
    file_path = join(data_dir, split+".char.bmes")
    with open(file_path, 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    return word_lists, tag_lists

def build_vocab(word_lists, tag_lists):
    word2id = _build_map(word_lists)
    tag2id = _build_map(tag_lists)
    return word2id, tag2id

def _build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps
