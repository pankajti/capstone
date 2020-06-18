import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import pandas as pd
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

import tensorflow_hub as hub
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split


max_words = 15000
tokenizer = Tokenizer(num_words=max_words)
maxlen = 150
module_url = "https://tfhub.dev/google/elmo/3"
embed = hub.load(module_url)

def preprocess_text(text_eval):
    sequences = tokenizer.texts_to_sequences([text_eval])
    data = pad_sequences(sequences, maxlen=maxlen)
    return data

NAE_vals_array = np.empty((0, 1024), float)
PAE_vals_array = np.empty((0, 1024), float)


def abalation(args  ):
    index, row = args
    print("running for {}".format(index))

    global NAE_vals_array, PAE_vals_array
    model = load_model('../LSTM_Model.h5')

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


root_dir = r'/Users/pankaj/Library/Mobile Documents/com~apple~CloudDocs/Capstone/Wikipedia Data'

result_path = annot_file_path = os.path.join(root_dir , 'comments_with_grouped_annoptations.tsv')

merged_comments = pd.read_table(result_path)
merged_comments['recipient_attack'] = merged_comments['recipient_attack'].apply(lambda x : 1 if x> 1 else 0 )
X_train = merged_comments['comment']

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
sequences = tokenizer.texts_to_sequences(X_train)


data = pad_sequences(sequences, maxlen=maxlen)
print('Shape of data tensor:', data.shape)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
y = merged_comments['recipient_attack']

one_hot_train_labels = to_categorical(y)

labels = np.asarray(one_hot_train_labels)
print('Shape of label tensor:', labels.shape)
labels = labels[indices]


X_train, X_test_validate, y_train, y_test_validate = train_test_split(merged_comments.comment, merged_comments.recipient_attack,
                                                    stratify=labels,
                                                    test_size=0.40,
                                                    random_state=123)

DF_train = X_train.to_frame()
DF_train['recipient_attack'] = y_train

DF_train.reset_index(inplace=True)

from multiprocessing import Pool


def run_sequential ():
    for index, row in DF_train.iterrows():
        abalation((index , row))

def run_multiproc ():
    with Pool(10) as p:
        p.map(abalation , DF_train.iterrows() )


if __name__ == '__main__':
    run_multiproc()