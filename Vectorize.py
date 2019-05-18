import numpy as np
import jieba

word_list = np.load('Object/dict.npy')

word_list = word_list.tolist()

print(len(word_list))

print(word_list[-5:])

good = [line.strip().encode('utf8') for line in open('Files/good.txt', encoding='utf8').readlines()]
bad = [line.strip().encode('utf8') for line in open('Files/bad.txt', encoding='utf8').readlines()]
middle = [line.strip().encode('utf8') for line in open('Files/middle.txt', encoding='utf8').readlines()]

bad = bad + middle

print(len(bad))
print(len(good))
exit()

num_sentences = len(good) + len(bad)

max_seq_num = 36

ids = np.zeros((num_sentences, max_seq_num))

y_ids = []
count = 0

for item in bad:
    indexCounter = 0
    out1 = jieba.cut(item)
    for t in out1:
        try:
            ids[count][indexCounter] = word_list.index(t)
        except ValueError:
            ids[count][indexCounter] = -1
        indexCounter = indexCounter + 1
        if indexCounter >= max_seq_num:
            break
    count = count + 1

print(count)

for item in good:
    indexCounter = 0
    out1 = jieba.cut(item)
    for t in out1:
        try:
            ids[count][indexCounter] = word_list.index(t)
        except ValueError:
            ids[count][indexCounter] = -1
        indexCounter = indexCounter + 1
        if indexCounter >= max_seq_num:
            break
    count = count + 1

print(ids[-2:])

np.save('Object/idMatrix.npy', ids)

print("==== success ====")
