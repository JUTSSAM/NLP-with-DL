import numpy as np
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import sequence

ids = np.load('Object/idMatrix.npy')

print(ids.shape)

max_word = 36

x_train = ids
y_train = 38313 * [0] + 9684 * [1]

x_train = sequence.pad_sequences(x_train, maxlen=max_word)
# x_test = sequence.pad_sequences(x_test, maxlen=max_word)
vocab_size = np.max([np.max(x_train[i]) for i in range(x_train.shape[0])]) + 1

# 全连接多层神经网络
model = Sequential()
model.add(Embedding(vocab_size, 64, input_length=max_word))
model.add(Flatten())
# model.add(Dense(2000, activation='relu'))
# model.add(Dense(500, activation='relu'))
# model.add(Dense(200, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])




print(model.summary())

model.fit(x_train, y_train, epochs=2, batch_size=100, verbose=1)
model.save('Object/model.cpkt')
