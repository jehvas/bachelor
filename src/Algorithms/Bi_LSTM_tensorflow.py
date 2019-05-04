from Algorithms.AbstractTensorflowAlgorithm import AbstractTensorflowAlgorithm
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Embedding, Dense, LSTM, Activation, Bidirectional, Dropout
from utility.model_factory import make_hidden_layers


class Bi_LSTM_Tensorflow(AbstractTensorflowAlgorithm):
    def generate_model(self, middle_layers, input_shape):
        self.model = Sequential(make_hidden_layers(middle_layers, input_shape))
