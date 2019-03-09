#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 23:21:58 2018

@author: xiaodan
"""

import sys

sum_matrix = 0
sum_diag = 0
idx = 0
CM=[]
has_comma = False
for line in sys.stdin:
    if "," in line:
        has_comma = True
    line_strip = line.strip().strip("[ ]")
    if has_comma:
        line_split = map(lambda x: x.split()[0], line_strip.split(","))
    else:
        line_split = line_strip.split()
    if len(line_split) > 0:
        num_class = len(line_split)
        int_list = list(map(int, map(float, line_split)))
        sum_matrix += sum(int_list)
        sum_diag += int_list[idx]
        CM.append(int_list)
        
        idx += 1

    if idx == num_class:
        break

for i in range(len(CM)):
    TP=CM[i][i]
    FP=sum(x[i] for x in CM )-TP
    FN=sum(CM[i])-TP
    F1=2*TP/(2*TP+FP+FN)
    print ('F1 score for class ',i+1,' : ',F1)