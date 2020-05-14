from tensorflow import keras

char_idx = {chr(i) : i-ord('a') for i in range(ord('a'), ord('z')+1)}
char_idx[' '] = len(char_idx)
char_idx['!'] = len(char_idx)
char_idx['{'] = len(char_idx)


from gensim.models.poincare import PoincareModel

text = "my name is pankaj"


rev_char_idx = {c:i for i, c in char_idx.items()}

class HyperbolicEmbedding(keras.layers.Embedding) :

    def __init__(self):
        pass
    def build(self):
        pass
    def call(self):
        pass

import numpy as np
X= np.array([char_idx[ch] for ch in text])
y = np.array([char_idx[chr(ord(ch)+1)] for ch in text])

model  = keras.Sequential()
#model.add(HyperbolicEmbedding(len(char_idx),8,input_length=1))
model.add(Embedding(len(char_idx),8,input_length=1))
model.layers[0].set_weight(weights)
m,odel.layers[0].settea(false)
model.add(keras.layers.Dense(32))
model.add(keras.layers.LSTM(32))

model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1))
model.compile(optimizer="adam" , loss= 'mse')

model.summary()
history = model.fit(X, y, epochs= 220)
model.predict(char_idx['a'])
print(history)


