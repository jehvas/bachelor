import os
from typing import List

import numpy as np
import tensorflow as tf
from keras_preprocessing import sequence
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split as tts
from tensorflow import keras
from tensorflow.python.keras import Input, Model, Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Embedding, LSTM, Dense, Activation, Dropout, Bidirectional, RNN, \
    SimpleRNNCell
from tensorflow.python.keras.optimizers import RMSprop

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_name():
    return 'LSTM_Tensorflow'


def run_train(dataset, emails, labels, parameters) -> (List, List, List):
    x_train, x_test, y_train, y_test = tts(emails, labels, test_size=0.2, random_state=1, stratify=labels)

    output_dim = parameters['output_dim']
    hidden_dim = parameters['hidden_dim']
    input_dim = parameters['input_dim']

    max_words = 1000
    max_len = 150
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(x_train)
    sequences = tok.texts_to_sequences(x_train)
    sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)

    def AAS():
        inputs = Input(name='inputs', shape=[max_len])
        layer = Embedding(max_words, input_dim, input_length=max_len)(inputs)
        cell = SimpleRNNCell(64)
        layer = RNN(cell)(layer)
        layer = Dense(256, name='FC1')(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(output_dim, name='out_layer')(layer)
        layer = Activation('sigmoid')(layer)
        model = Model(inputs=inputs, outputs=layer)
        return model

    model = AAS()
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    #model.fit(x_train, np.asarray(y_train), epochs=20)
    history = model.fit(sequences_matrix, y_train, batch_size=128, epochs=5,
              validation_split=0.2)


    # predictions = model.predict(x_test)
    # predictions = [np.argmax(x) for x in predictions]

    test_sequences = tok.texts_to_sequences(x_test)
    test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)

    accr = model.evaluate(test_sequences_matrix, y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
    return [], y_test, []
