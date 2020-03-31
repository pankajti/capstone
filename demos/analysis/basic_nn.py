X= [[1,2,3],[ 10,20,30],[100,200,300],[100,200,300]]
y= [[1,2],[2,4],[3,4],[1,3]]
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.callbacks import TensorBoard
tf.keras.callbacks.TensorBoard()
model = Sequential()
model.add(Dense(10, activation ='sigmoid'))
model.add(Dense(2))
tb = TensorBoard(log_dir = './logs/basic_nn')
model.compile(optimizer = "rmsprop", loss="categorical_crossentropy", metrics = ['acc'])

history = model.fit(X,y, epochs =1, batch_size= 1)
#model.summary()
