from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow import keras

'''
Input preparation
'''
tf.random.set_seed(1234)

X = tf.ones(shape=(10,2))
y= tf.zeros(shape=(10))


model  = Sequential()
layer1  = Dense(2,activation = "relu")
model.add(layer1)
model.compile(optimizer= "adam" , loss= "binary_crossentropy")
history = model.fit(X ,y )


print(model.layers[0].weights[0])

print(model.predict(tf.ones(shape = (1,2))))


mat_mul = tf.matmul(tf.ones(shape = (1,2)) , model.layers[0].weights[0] )
print(mat_mul)
print(keras.activations.relu(mat_mul))