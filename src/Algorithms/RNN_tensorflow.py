import numpy as np
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Dense, RNN, SimpleRNNCell, Dropout, LeakyReLU

from Algorithms.AbstractTensorflowAlgorithm import AbstractTensorflowAlgorithm


class RNN_Tensorflow(AbstractTensorflowAlgorithm):
    def prepare_features(self, x_train, y_train, x_test, y_test):
        x_train = np.expand_dims(x_train, axis=1)
        y_train = np.expand_dims(y_train, axis=1)
        x_test = np.expand_dims(x_test, axis=1)
        y_test = np.expand_dims(y_test, axis=1)
        return x_train, y_train, x_test, y_test

    def generate_model(self, hidden_layers, input_shape, output_dim):
        layers = []
        layers.append(RNN(SimpleRNNCell(hidden_layers[0][1], activation=hidden_layers[0][2]), input_shape=input_shape, return_sequences=True))
        if hidden_layers[0][2] == 'linear':
            layers.append(LeakyReLU())
        layers.append(Dropout(hidden_layers[1][1]))
        layers.append(RNN(SimpleRNNCell(hidden_layers[2][1])))
        if hidden_layers[2][2] == 'linear':
            layers.append(LeakyReLU())
        layers.append(Dropout(hidden_layers[3][1]))
        layers.append(Dense(hidden_layers[4][1], activation=hidden_layers[4][2]))
        if hidden_layers[4][2] == 'linear':
            layers.append(LeakyReLU())
        layers.append(Dropout(hidden_layers[5][1]))
        layers.append(Dense(output_dim, activation='softmax'))
        self.model = Sequential(layers)
