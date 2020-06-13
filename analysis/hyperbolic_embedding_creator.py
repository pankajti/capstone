from tensorflow.keras.layers import LSTM
import numpy as np
embeddings_index = {}

embedding_dim = 16
with open('../word2_wec_poinc', 'r') as f:
    #emb= f.readlines()


    for line in f:
        values = line.split()
        word = values[0]
        w= word.split(".")[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[w] = coefs

print('Found %s word vectors.' % len(embeddings_index))
