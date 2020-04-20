from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential


model  = Sequential()
model.add(Dense(2,activation = "relu"))
model.compile(optimizer= "adam" , loss= "binary_crossentropy")
history = model.fit(tf.ones(shape=(10,2)) , tf.zeros(shape=(10)))
model.layers[0].weights
print(history)



