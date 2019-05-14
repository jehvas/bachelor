import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, GlobalAveragePooling1D, LeakyReLU

from Algorithms.AbstractTensorflowAlgorithm import AbstractTensorflowAlgorithm
from utility.model_factory import make_hidden_layers


class MLP_Tensorflow(AbstractTensorflowAlgorithm):
    def prepare_features(self, x_train, y_train, x_test, y_test):
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        return x_train, y_train, x_test, y_test

    def generate_model(self, middle_layers, input_shape):
        self.model = Sequential(make_hidden_layers(middle_layers, input_shape))
