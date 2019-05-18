#!/usr/bin/python3
# -*- coding:utf8 -*-
corpus = [line.strip().encode('utf8') for line in open('Source/all.txt', encoding='utf8').readlines()]
for item in corpus:
    sentence = item.split()
    if len(sentence) < 2:
        print(sentence)
        continue

    print(sentence[0].decode('utf8'))
    print(float(sentence[1]))
    if float(sentence[1]) > 0.5:
        with open('Files/good.txt', 'a', encoding='utf-8') as f1:
            f1.write((sentence[0]).decode('utf-8') + '\n')
    else:
        if float(sentence[1]) < 0.5:
            with open('Files/bad.txt', 'a', encoding='utf-8') as f2:
                f2.write((sentence[0]).decode('utf-8') + '\n')
        else:
            with open('Files/middle.txt', 'a', encoding='utf-8') as f3:
                f3.write((sentence[0]).decode('utf-8') + '\n')
print(" === success ===")
f1.close()
f2.close()
f3.close()
