import numpy as np
from tensorflow.python.keras import Input

from Algorithms.AbstractTensorflowAlgorithm import AbstractTensorflowAlgorithm
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Embedding, Dense, LSTM, Activation, Bidirectional, Dropout, CuDNNLSTM, \
    LeakyReLU
import tensorflow as tf

class Bi_LSTM_Tensorflow(AbstractTensorflowAlgorithm):
    def prepare_features(self, x_train, y_train, x_test, y_test):
        x_train = np.expand_dims(x_train, axis=1)
        y_train = np.expand_dims(y_train, axis=1)
        x_test = np.expand_dims(x_test, axis=1)
        y_test = np.expand_dims(y_test, axis=1)
        return x_train, y_train, x_test, y_test

    def generate_model(self, hidden_layers, input_shape, output_dim):
        if not tf.test.gpu_device_name():
            raise Exception("This device does not support the GPU version of Tensorflow.\nPlease install needed "
                            "drivers or run on Google Colab")
        layers = []
        layers.append(Input(input_shape))
        layers.append(Bidirectional(CuDNNLSTM(hidden_layers[0][1], return_sequences=True)))
        if hidden_layers[0][2] != 'linear':
            layers.append(LeakyReLU())
        layers.append(Dropout(hidden_layers[1][1]))
        layers.append(Bidirectional(CuDNNLSTM(hidden_layers[2][1])))
        if hidden_layers[2][2] != 'linear':
            layers.append(LeakyReLU())
        layers.append(Dropout(hidden_layers[3][1]))
        layers.append(Dense(hidden_layers[4][1], activation=hidden_layers[4][2]))
        if hidden_layers[4][2] != 'linear':
            layers.append(LeakyReLU())
        layers.append(Dropout(hidden_layers[5][1]))
        layers.append(Dense(output_dim, activation='softmax'))

        self.model = Sequential(layers)