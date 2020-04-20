from tensorflow import keras
import tensorflow as tf

model = keras.Sequential()

model.add(keras.layers.Embedding(2,1))
model.compile(optimizer='rmsprop', loss = 'mse')
model.fit(tf.ones(shape = (3,2)), tf.ones(shape = (3)))

print(model)