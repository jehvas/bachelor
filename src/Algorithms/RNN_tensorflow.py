from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Embedding, Dense, StackedRNNCells, RNN, SimpleRNNCell, CuDNNGRU, Dropout

from Algorithms.AbstractTensorflowAlgorithm import AbstractTensorflowAlgorithm
from utility.model_factory import make_hidden_layers


class RNN_Tensorflow(AbstractTensorflowAlgorithm):
    def generate_model(self, hidden_layers, input_shape):
        self.model = Sequential(make_hidden_layers(hidden_layers, input_shape))
