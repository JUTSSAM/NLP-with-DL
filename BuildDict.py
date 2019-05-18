#!/usr/bin/python3
# -*- coding:utf8 -*-
import jieba
import numpy as np

good = [line.strip().encode('utf8') for line in open('Files/good.txt', encoding='utf8').readlines()]
bad = [line.strip().encode('utf8') for line in open('Files/bad.txt', encoding='utf8').readlines()]
middle = [line.strip().encode('utf8') for line in open('Files/middle.txt', encoding='utf8').readlines()]

texts = good + bad + middle

out1 = []
for item in texts:
    re = jieba.cut(item)
    out1 = out1 + list(re)

out1 = list(set(out1))

out1.sort(reverse=True)

dict(list(zip(out1, np.arange(len(out1)))))

np.save('Object/dict.npy', out1)

print(" === success ===")
