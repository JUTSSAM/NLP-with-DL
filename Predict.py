from keras.models import load_model
import numpy as np
import jieba
from keras.preprocessing import sequence

model = load_model('Object/model.cpkt')

word_list = np.load('Object/dict.npy')
word_list = word_list.tolist()

txt = '捡冠军?笑死我了，那你RNG怎么也不捡一个?'

out1 = jieba.cut(txt)
out1 = list(out1)
data = [0] * len(txt)
count = 0
for item in out1:
    try:
        data[count] = word_list.index(item)
    except ValueError:
        data[count] = -1
    count = count + 1

input_data = sequence.pad_sequences([data], 36)

pred = model.predict(input_data)
