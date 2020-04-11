from tensorflow.keras.layers import Dense,SimpleRNN
from tensorflow.keras import Sequential
import numpy as np
from tensorflow.keras.utils import plot_model


model =Sequential()

model.add(Dense(2))
model.add(Dense(1))

plot_model(model)

