import numpy as np
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
from keras.layers import Dense
from keras.datasets import imdb
from keras.preprocessing import sequence

(x_train, y_train), (x_test, y_test) = imdb.load_data()

max_word = 400
x_train = sequence.pad_sequences(x_train, maxlen=max_word)
x_test = sequence.pad_sequences(x_test, maxlen=max_word)
vocab_size = np.max([np.max(x_train[i]) for i in range(x_train.shape[0])]) + 1

model = Sequential()
model.add(Embedding(vocab_size, 64, input_length=max_word))
model.add(Flatten())
model.add(Dense(2000, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=100, verbose=1)
score = model.evaluate(x_test, y_test)
model.save('Object/model.cpkt')
