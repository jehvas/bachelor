import os
from typing import List, Counter

import numpy as np
import tensorflow as tf
from keras_preprocessing import sequence
from sklearn.model_selection import train_test_split as tts
from tensorflow import keras
from tensorflow.python.keras import Input, Model, Sequential
from tensorflow.python.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.python.keras.layers import Embedding, LSTM, Dense, Activation, Dropout, Bidirectional, CuDNNGRU, GRU, \
    RNN, SimpleRNNCell
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.ops.rnn_cell_impl import RNNCell

from utility.model_factory import generate_model
from utility.plotter import PlotClass


def learning_rate_function(epoch, learning_rate):
    return learning_rate * 0.99


def get_name():
    return 'RNN_Tensorflow'


def run_train(dataset, features, labels, parameters, embedding=None) -> (List, List, List):
    x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2, random_state=1, stratify=labels)

    output_dim = parameters['output_dim']
    hidden_dim = parameters['hidden_dim']
    input_dim = parameters['input_dim']
    num_epochs = parameters['num_epochs']
    batch_size = parameters['batch_size']
    input_function = parameters['input_function']
    hidden_layers = parameters['hidden_layers']
    output_function = parameters['output_function']
    optimizer = parameters['optimizer']
    loss_function = parameters['loss_function']

    def RNN_model():
        if tf.test.is_gpu_available():
            rnn = CuDNNGRU
        else:
            import functools
            rnn = functools.partial(GRU, recurrent_activation='sigmoid')

        model = Sequential([
            Embedding(embedding.shape[0], embedding.shape[1], weights=embedding),
            Dropout(0.5),
            RNN(SimpleRNNCell(hidden_dim)),
            # Dense(hidden_dim, name='FC1', activation=input_function),
            # rnn(rnn_units, recurrent_initializer='glorot_uniform'),
            Dense(output_dim, name='out_layer', activation=output_function),
        ])

        '''
        # model = generate_model(input_dim, hidden_dim, hidden_layers, output_dim, input_function, output_function, isRNN=True)
        inputs = Input(name='inputs', shape=[input_dim])
        layer = Embedding(embedding.shape[0], embedding.shape[1], batch_input_shape=[batch_size, 256], weights=embedding)(inputs),
        cell = SimpleRNNCell(hidden_dim)
        layer = RNN(cell)(layer)
        layer = Dense(hidden_dim, name='FC1')(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(output_dim, name='out_layer')(layer)
        layer = Activation('sigmoid')(layer)
        model = Model(inputs=inputs, outputs=layer)
        '''
        return model

    rnn_model = RNN_model()
    rnn_model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
    rnn_model.summary()
    history = rnn_model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
                            validation_data=(x_test, y_test), workers=12,
                            callbacks=[LearningRateScheduler(learning_rate_function, verbose=1),
                                       EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1,
                                                     mode='auto',
                                                     restore_best_weights=True)
                                       ])

    iteration_list = [i for i in range(1, num_epochs + 1)]

    predictions = rnn_model.predict(x_test)
    predictions = [np.argmax(x) for x in predictions]

    accr = rnn_model.evaluate(x_test, y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
    return ([
        PlotClass([(iteration_list, history.history['val_accuracy'])], "Epoch", "Accuracy", parameters, dataset, "RNN",
                  legend=(['train', 'test'], 'upper left')),
        PlotClass([(iteration_list, history.history['val_loss'])], "Epoch", "Loss", parameters, dataset, "RNN",
                  legend=(['train', 'test'], 'upper left'))
    ]), y_test, predictions
