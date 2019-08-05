#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : main.py
# Create date : 2019-08-02 17:08
# Modified date : 2019-08-03 17:35
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

from run_models import run

def main():
    run(train=True)
    run(train=False)

if __name__ == "__main__":
    main()
