#!/usr/bin/env python
# coding: utf-8

# In[7]:


import tensorflow as tf


# In[8]:


tf.__version__


# In[9]:


import tensorflow as tf
if tf.test.gpu_device_name():
   print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")


# In[10]:


from sklearn import *
from tensorflow import keras as keras
# import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import load_digits
from tensorflow.keras.utils import to_categorical
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
from tensorboard.plugins import projector

from sklearn.model_selection import train_test_split


# In[11]:


#root_dir = 'C:\\Users\\Shantanu\\Dropbox\\Capstone\\Wikipedia Data\\'

root_dir = r'/Users/pankaj/Library/Mobile Documents/com~apple~CloudDocs/Capstone/Wikipedia Data'

result_path = annot_file_path = os.path.join(root_dir , 'comments_with_grouped_annoptations.tsv')

merged_comments = pd.read_table(result_path)


# In[12]:


merged_comments['recipient_attack'] = merged_comments['recipient_attack'].apply(lambda x : 1 if x> 1 else 0 )
X_train = merged_comments['comment']
y = merged_comments['recipient_attack']
one_hot_train_labels = to_categorical(y)


# In[13]:


merged_comments['new_attack'] = merged_comments['attack'].apply(lambda x : 1 if x> 1 else 0 )


# In[14]:


# Prepare training input  
training_samples = 0.60
validation_samples = 0.40
max_words = 15000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
sequences = tokenizer.texts_to_sequences(X_train)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# In[15]:


maxlen = 150


# In[16]:


data = pad_sequences(sequences, maxlen=maxlen)
print('Shape of data tensor:', data.shape)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
# x_train = data[:training_samples]
# x_val = data[training_samples: (training_samples + validation_samples)]
# x_test = data[(training_samples + validation_samples) : ]


# In[17]:


# Prepare labels 
labels = np.asarray(one_hot_train_labels)
print('Shape of label tensor:', labels.shape)
labels = labels[indices]
# y_train = labels[:training_samples]
# y_val = labels[training_samples: training_samples + validation_samples]
# y_test = labels[(training_samples + validation_samples) : ]


# In[18]:


#### Stratified Test Train Validation Split 

### First Lets do 60 and 40. 

X_train, X_test_validate, y_train, y_test_validate = train_test_split(data, labels,
                                                    stratify=labels, 
                                                    test_size=0.40,
                                                    random_state=123)



# In[19]:


### Then 20% for Testing and Validation 

X_test, X_validate, y_test, y_validate = train_test_split(X_test_validate, y_test_validate,
                                                    stratify = y_test_validate, 
                                                    test_size=0.50,
                                                    random_state=123)


# In[20]:


len(X_train)+len(X_test)+len(X_validate)


# In[21]:


ES = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)


# In[22]:


from tensorflow.keras.layers import LSTM
embedding_dim = 100
model = Sequential()
#embedding = Embedding(max_words, embedding_dim , weights =w)
embedding = Embedding(max_words, embedding_dim)

#embedding.trainable = False
model.add(embedding)
model.add(LSTM(128,return_sequences = True))
model.add(LSTM(64))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.summary()


# In[ ]:


model.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['acc'])
history = model.fit(X_train, y_train,
epochs= 12,
batch_size=256,
callbacks = [ES],
validation_data=(X_validate, y_validate))


# In[ ]:


model.save('LSTM_Model.h5')


# In[ ]:


from tensorflow.keras.models import load_model


# In[ ]:


model1 = load_model('LSTM_Model.h5')


# In[ ]:


X_test[1].shape


# In[ ]:


X_test.shape


# In[ ]:


test_output = model1.predict([X_test])


# In[ ]:


test_output.shape


# In[ ]:


model1.summary()


# In[ ]:


y_pred = np.np.argmax([test_output])


# In[ ]:


y_pred


# In[ ]:


test_output[0:5]


# In[ ]:


y_pred = list(map(lambda x: np.argmax(x), test_output))
y_test_bin = list(map(lambda x: np.argmax(x), y_test))


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


cfm = confusion_matrix(y_test_bin,y_pred)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


print(accuracy_score(y_test_bin,y_pred))


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


report = classification_report(y_test_bin,y_pred)


# In[ ]:


print(report)


# In[ ]:


from sklearn.metrics import plot_confusion_matrix


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


class_names=["Non-Toxic","Toxic"] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cfm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[ ]:


import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 


# In[ ]:


#### Testing ELMO

import tensorflow_hub as hub

module_url = "https://tfhub.dev/google/elmo/3"
embed = hub.load(module_url)


# In[ ]:


merged_comments.head()


# In[ ]:


X_train, X_test_validate, y_train, y_test_validate = train_test_split(merged_comments.comment, merged_comments.recipient_attack,
                                                    stratify=labels, 
                                                    test_size=0.40,
                                                    random_state=123)


# In[ ]:


DF_train = X_train.to_frame()


# In[ ]:


DF_train['recipient_attack'] = y_train


# In[ ]:


DF_train.head()


# In[ ]:


DF_train.reset_index(inplace=True)


# In[ ]:


DF_train.head()


# In[ ]:


def preprocess_text(text_eval):
    sequences = tokenizer.texts_to_sequences([text_eval])
    data = pad_sequences(sequences, maxlen=maxlen)
    return data


# In[ ]:


DF_train.index[0:10]


# In[ ]:


NAE_vals_array = np.empty((0,1024), float)
PAE_vals_array = np.empty((0,1024), float)


# In[ ]:


NAE_vals_array


# In[ ]:


PAE_vals_array


# In[ ]:


### Abalation and Get the Vectors of the important trigrams 

def abalation():
    global NAE_vals_array, PAE_vals_array
    NAE_vals_array = np.empty((0, 1024), float)
    PAE_vals_array = np.empty((0, 1024), float)
    for index, row in DF_train.iterrows():

        text = row.comment.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        tokens_without_sw = [word for word in tokens if not word in stopwords.words()]

        ### First Get the overall Classification

        pred_vlaues = model.predict(preprocess_text(str(tokens_without_sw)))

        classification = pred_vlaues.argmax()

        if classification == 0 and row.recipient_attack == 0:
            mode = 'PAE'  # Positive AutoEncoder
        elif classification == 0 and row.recipient_attack == 1:
            mode = 'NAE'
        elif classification == 1 and row.recipient_attack == 0:
            mode = 'PAE'
        elif classification == 1 and row.recipient_attack == 1:
            mode = 'NAE'

        ### Get the Elmo vector

        text = tf.convert_to_tensor([str(tokens_without_sw)])
        out = embed.signatures['default'](text)['elmo']

        pred_vals_array = np.empty((0, 2), float)

        for i in range(len(tokens_without_sw) - 2):
            text_eval = tokens_without_sw[i:i + 3]

            a = preprocess_text(str(text_eval))
            pred_vlaues = model.predict(a)
            pred_vals_array = np.vstack((pred_vals_array, pred_vlaues))

        if mode == "NAE":

            # Only Negative Vals, and find the index of the highest contributing phrase

            phrase_start_index = pred_vals_array[:, 1].argmax()

            ### Get the ELMO Encoding of the entire vector:

            elmo_vecs = np.array(out[0][phrase_start_index:phrase_start_index + 3])

            NAE_vals_array = np.vstack((NAE_vals_array, elmo_vecs))

            print("NAE")
            print(len(NAE_vals_array))

        elif mode == "PAE":

            # Only Positive Vals, and find the index of the highest contributing phrase

            phrase_start_index = pred_vals_array[:, 0].argmax()

            PAE_vals_array = np.vstack((PAE_vals_array, out[0][phrase_start_index:phrase_start_index + 3]))

            print("PAE")
            print(len(PAE_vals_array))

        if index == 20:
            break


adalation()

# In[ ]:


NAE_vals_array.shape


# In[ ]:


PAE_vals_array.shape


# In[ ]:


NAE_vals_array = NAE_vals_array.reshape(int(NAE_vals_array.shape[0]/3),3,1024)
PAE_vals_array = PAE_vals_array.reshape(int(PAE_vals_array.shape[0]/3),3,1024)


# In[ ]:


PAE_vals_array.shape


# In[ ]:


# lstm autoencoder to recreate a timeseries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:


timesteps = 3
n_features = 1024
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

# define model
model_PAE = Sequential()
model_PAE .add(LSTM(1024, activation='relu', input_shape=(timesteps,n_features), return_sequences=True))
model_PAE .add(LSTM(720, activation='relu', input_shape=(timesteps,n_features), return_sequences=True))
model_PAE .add(LSTM(128, activation='relu', input_shape=(timesteps,n_features), return_sequences=True))
model_PAE .add(LSTM(64, activation='relu', return_sequences=False))
model_PAE .add(RepeatVector(timesteps))
model_PAE .add(LSTM(64, activation='relu', return_sequences=True))
model_PAE .add(LSTM(128, activation='relu', return_sequences=True))
model_PAE .add(LSTM(720, activation='relu', return_sequences=True))
model_PAE .add(LSTM(1027, activation='relu', return_sequences=True))
model_PAE .add(TimeDistributed(Dense(n_features)))
model_PAE .compile(optimizer='adam', loss='mse')
model_PAE.summary()


# In[ ]:


model_PAE.fit(PAE_vals_array, PAE_vals_array, epochs=3, batch_size=5)


# In[ ]:





# In[ ]:


timesteps = 3
n_features = 1024
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

# define model
model_NAE = Sequential()
model_NAE .add(LSTM(1024, activation='relu', input_shape=(timesteps,n_features), return_sequences=True))
model_NAE .add(LSTM(720, activation='relu', input_shape=(timesteps,n_features), return_sequences=True))
model_NAE .add(LSTM(128, activation='relu', input_shape=(timesteps,n_features), return_sequences=True))
model_NAE .add(LSTM(64, activation='relu', return_sequences=False))
model_NAE .add(RepeatVector(timesteps))
model_NAE .add(LSTM(64, activation='relu', return_sequences=True))
model_NAE .add(LSTM(128, activation='relu', return_sequences=True))
model_NAE .add(LSTM(720, activation='relu', return_sequences=True))
model_NAE .add(LSTM(1027, activation='relu', return_sequences=True))
model_NAE .add(TimeDistributed(Dense(n_features)))
model_NAE .compile(optimizer='adam', loss='mse')
model_NAE.summary()


# In[ ]:


model_NAE.fit(NAE_vals_array, NAE_vals_array, epochs=3, batch_size=5)

