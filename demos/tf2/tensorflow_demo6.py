import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
import numpy as np

model = keras.Sequential()

model.add(keras.layers.Dense(1, activation = "linear"))
model.compile(optimizer="adam", loss="binary_crossentropy")

X = tf.ones(shape=(12,1))
print(X)
y = tf.ones(shape=(12))
print(y)

history = model.fit(X, y , epochs=20)
plot_model(model)

print(model.layers[0].weights[0])

print(model.predict(tf.ones(shape = (1,1))))

#model.summary()


from sklearn.linear_model import LinearRegression

reg  = LinearRegression()
X= [[1],[1],[1]]
reg.fit(X, [1,1,1])

print(reg.predict([[1]]))