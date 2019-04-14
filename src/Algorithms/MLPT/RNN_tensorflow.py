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

from utility.model_factory import generate_model
from utility.plotter import PlotClass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_name():
    return 'RNN_Tensorflow'


def run_train(dataset, features, labels, parameters, emails) -> (List, List, List):
    x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2, random_state=1, stratify=labels)

    '''
    output_dim = parameters['output_dim']
    hidden_dim = parameters['hidden_dim']
    input_dim = parameters['input_dim']
    max_len = parameters['max_len']
    dropout = parameters['dropout']
    num_epochs = parameters['num_epochs']
    batch_size = parameters['batch_size']
    '''

    output_dim = parameters['output_dim']
    hidden_dim = parameters['hidden_dim']
    input_dim = parameters['input_dim']
    # max_len = parameters['max_len']
    num_epochs = parameters['num_epochs']
    batch_size = parameters['batch_size']
    input_function = parameters['input_function']
    middle_layers = parameters['middle_layers']
    output_function = parameters['output_function']

    def RNN_model():
        # model = generate_model(input_dim, hidden_dim, middle_layers, output_dim, input_function, output_function,
        #                       isRNN=True)

        inputs = Input(name='inputs', shape=[input_dim])
        # layer = Embedding(len(matrix), input_dim, weights=[matrix], trainable=False, input_length=max_len)(inputs)
        cell = SimpleRNNCell(hidden_dim)
        layer = RNN(cell)(inputs)
        layer = Dense(hidden_dim, name='FC1')(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(output_dim, name='out_layer')(layer)
        layer = Activation('sigmoid')(layer)
        model = Model(inputs=inputs, outputs=layer)

        return model

    rnn_model = RNN_model()
    rnn_model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    history = rnn_model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
                            validation_data=(x_test, y_test))

    iteration_list = [i for i in range(1, num_epochs + 1)]

    predictions = rnn_model.predict(x_test)
    rounded_predictions = [np.argmax(x) for x in predictions]

    accr = rnn_model.evaluate(x_test, y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
    return ([
        PlotClass([(iteration_list, history.history['val_acc'])], "Epoch", "Accuracy", parameters, dataset, "RNN",
                  legend=(['train', 'test'], 'upper left')),
        PlotClass([(iteration_list, history.history['val_loss'])], "Epoch", "Loss", parameters, dataset, "RNN",
                  legend=(['train', 'test'], 'upper left'))
    ]), y_test, rounded_predictions
