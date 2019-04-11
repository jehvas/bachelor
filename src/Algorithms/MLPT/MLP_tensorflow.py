from typing import List

import keras
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split as tts
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Flatten, Dense, Activation, Dropout, Embedding, ELU, PReLU, ReLU, Softmax
from tensorflow.python.keras.optimizers import Adam

from utility.plotter import PlotClass


def get_name():
    return 'MLP_Tensorflow'


def run_train(dataset, features, labels, parameters, matrix, sequences_matrix, emails) -> (List, List, List):
    data = None
    if features is not None:
        data = features
    elif matrix is not None:
        data = matrix
    x_train, x_test, y_train, y_test = tts(data, labels, test_size=0.2, random_state=1, stratify=labels)

    output_dim = parameters['output_dim']
    hidden_dim = parameters['hidden_dim']
    input_dim = parameters['input_dim']
    # max_len = parameters['max_len']
    dropout = parameters['dropout']
    num_epochs = parameters['num_epochs']
    batch_size = parameters['batch_size']
    input_function = parameters['input_function']
    hidden_layers = parameters['hidden_layers']
    output_function = parameters['output_function']
    use_dropout = parameters['use_dropout']

    def MLP():

        layers = [tf.keras.layers.Dense(features.shape[1], activation=input_function)] + \
                 ([tf.keras.layers.Dropout(dropout)] if use_dropout else []) + \
                 hidden_layers + \
                 [tf.keras.layers.Dense(output_dim, activation=output_function)]
        model = tf.keras.Sequential(layers)

        '''model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation=tf.nn.softplus),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(hidden_dim, activation=tf.nn.softsign),
            tf.keras.layers.Dense(output_dim, activation=tf.nn.tanh)
        ])'''

        return model

    mlp_model = MLP()
    mlp_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    history = mlp_model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
                            validation_data=(x_test, y_test), workers=4, verbose=0)

    iteration_list = [i for i in range(1, num_epochs + 1)]

    predictions = mlp_model.predict(x_test)
    rounded_predictions = [np.argmax(x) for x in predictions]

    accr = mlp_model.evaluate(x_test, y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
    return ([
        PlotClass([(iteration_list, history.history['val_accuracy'])], "Epoch", "Accuracy", parameters, dataset, "MLP",
                  legend=(['train', 'test'], 'upper left')),
        PlotClass([(iteration_list, history.history['val_loss'])], "Epoch", "Loss", parameters, dataset, "MLP",
                  legend=(['train', 'test'], 'upper left'))
    ]), y_test, rounded_predictions
