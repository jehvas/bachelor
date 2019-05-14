import numpy as np
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Embedding, Dense, StackedRNNCells, RNN, SimpleRNNCell, CuDNNGRU, Dropout

from Algorithms.AbstractAlgorithm import AbstractAlgorithm
from Algorithms.AbstractTensorflowAlgorithm import AbstractTensorflowAlgorithm
from utility.model_factory import make_hidden_layers


class RNN_Tensorflow(AbstractTensorflowAlgorithm):
    def prepare_features(self, x_train, y_train, x_test, y_test):
        x_train = np.expand_dims(x_train, axis=1)
        y_train = np.expand_dims(y_train, axis=1)
        x_test = np.expand_dims(x_test, axis=1)
        y_test = np.expand_dims(y_test, axis=1)
        return x_train, y_train, x_test, y_test

    def generate_model(self, hidden_layers, input_shape):
        self.model = Sequential(make_hidden_layers(hidden_layers, input_shape))
