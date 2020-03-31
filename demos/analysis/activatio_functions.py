from tensorflow.keras.datasets import imdb
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding,Flatten
from tensorflow.keras import preprocessing
import tensorflow as tf

max_features = 10000
max_len = 20
(x_train , y_train), (x_test,y_test)= imdb.load_data(num_words=max_features)
x_train = preprocessing.sequence.pad_sequences(x_train , maxlen = max_len)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

model = Sequential()
model.add(Embedding(10000, 8, input_length=max_len ))
model.add(Flatten())
model.add(Dense(1, activation='softmax'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
tf_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')
history = model.fit(x_train, y_train , epochs= 10 , batch_size= 32, validation_split = 0.2, callbacks =[tf_callback])
