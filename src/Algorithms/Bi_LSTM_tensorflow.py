from Algorithms.AbstractTensorflowAlgorithm import AbstractTensorflowAlgorithm
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Embedding, Dense, LSTM
from utility.model_factory import make_hidden_layers


class Bi_LSTM_Tensorflow(AbstractTensorflowAlgorithm):
    def generate_model(self):
        '''
        self.model = Sequential(
            [Embedding(self.embedding.shape[0], self.embedding.shape[1], weights=[self.embedding],
                       input_length=self.input_dim, trainable=False)] +
            make_hidden_layers(self.hidden_dim, self.hidden_layers) +
            [Dense(self.output_dim, name='out_layer')]
        )
        '''
        self.model = Sequential(
            [Embedding(self.embedding.shape[0], self.embedding.shape[1], weights=[self.embedding],
                       input_length=self.input_dim, trainable=False),
             LSTM(128),
             Dense(20, activation='softmax')])
