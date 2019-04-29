import tensorflow as tf
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Dense

from Algorithms.AbstractTensorflowAlgorithm import AbstractTensorflowAlgorithm
from utility.model_factory import make_hidden_layers


class MLP_Tensorflow(AbstractTensorflowAlgorithm):
    def generate_model(self):
        self.model = Sequential(
            [Dense(self.hidden_dim, input_shape=(self.input_dim,), activation=self.input_function)] +
            make_hidden_layers(self.hidden_dim, self.hidden_layers) +
            [Dense(self.output_dim, name='out_layer', activation=self.output_function)]
        )

