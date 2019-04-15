import os
from typing import List

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split as tts
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Embedding, LSTM, Dense, Activation, Dropout, Bidirectional, LSTMCell
from tensorflow.python.keras.optimizers import RMSprop

from utility.model_factory import generate_model
from utility.plotter import PlotClass


def get_name() -> str:
    return 'Bi-LSTM_Tensorflow'


def run_train(dataset, features, labels, parameters, emails) -> (List, List, List):
    x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2, random_state=1, stratify=labels)

    # x_train = tf.reshape(x_train, [x_train.shape[0], x_train.shape[1], 1])
    # y_train = tf.reshape(y_train, [1, y_train.shape[0]])
    # y_train = tf.convert_to_tensor(y_train, np.float32)
    # x_test = tf.reshape(x_test, [x_test.shape[0], x_test.shape[1], 1])
    # y_test = tf.convert_to_tensor(y_test, np.float32)
    # y_test = tf.reshape(y_test, [1, y_test.shape[0]])

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

    def Bi_LSTM():
        max_len = 200
        '''
        model = generate_model(input_dim, hidden_dim, middle_layers, output_dim, input_function, output_function,
                               isLSTM=True)
        inputs = Input(name='inputs', shape=[max_len])
        # layer = Embedding(len(features), input_dim, weights=[features], trainable=False, input_length=max_len)(inputs)
        layer = LSTM(max_len)(inputs)
        layer = Bidirectional(layer)
        layer = Dense(hidden_dim, name='FC1')(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(output_dim, name='out_layer')(layer)
        layer = Activation('sigmoid')(layer)
        model = Model(inputs=inputs, outputs=layer)
        
        '''

        model = LSTMCell(max_len)
        return model

    bi_lstm_model = Bi_LSTM()
    # bi_lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    # history = bi_lstm_model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_test, y_test), workers=4)

    iteration_list = [i for i in range(1, num_epochs + 1)]

    predictions = bi_lstm_model.predict(x_test)
    rounded_predictions = [np.argmax(x) for x in predictions]

    accr = bi_lstm_model.evaluate(x_test, y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
    return ([
        PlotClass([(iteration_list, history.history['val_acc'])], "Epoch", "Accuracy", parameters, dataset, "Bi-LSTM",
                  legend=(['train', 'test'], 'upper left')),
        PlotClass([(iteration_list, history.history['val_loss'])], "Epoch", "Loss", parameters, dataset, "Bi-LSTM",
                  legend=(['train', 'test'], 'upper left'))
    ]), y_test, rounded_predictions
