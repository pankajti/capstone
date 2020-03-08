#!/usr/bin/env python
# coding: utf-8

# In[21]:


from keras.layers import Embedding
from keras import Sequential 
from keras.layers import Flatten , Dense
from tensorflow.keras.datasets import imdb
from keras import preprocessing


# In[24]:


max_features = 10000
max_len = 20
embedding_layer = Embedding(1000, 64)


# In[26]:


(x_train , y_train), (x_test,y_test)= imdb.load_data(num_words=max_features)
x_train = preprocessing.sequence.pad_sequences(x_train , maxlen = max_len)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)


# In[27]:


x_train.shape


# In[20]:


x_train[24999][0]


# In[28]:


model = Sequential ()
model.add(Embedding(10000, 8, input_length=max_len ))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()


# In[13]:


history = model.fit(x_train, y_train , epochs= 10 , batch_size= 32, validation_split = 0.2)


# In[39]:


import os 
imdb_dir = '/Users/pankaj/dev/git/smu/ML2/data/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')
labels = []
texts = []
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
for fname in os.listdir(dir_name):
    if fname[-4:] == '.txt':
        f = open(os.path.join(dir_name, fname))
        texts.append(f.read())
        f.close()
        if label_type == 'neg':
            labels.append(0)
        else:
            labels.append(1)


# In[ ]:





# In[ ]:





# In[ ]:




