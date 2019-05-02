from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Embedding, Dense, StackedRNNCells, RNN, SimpleRNNCell, CuDNNGRU, Dropout

from Algorithms.AbstractTensorflowAlgorithm import AbstractTensorflowAlgorithm
from utility.model_factory import make_hidden_layers


class RNN_Tensorflow(AbstractTensorflowAlgorithm):
    def generate_model(self):
        '''
        self.model = Sequential(
            [Embedding(self.embedding.shape[0], self.embedding.shape[1], weights=[self.embedding], trainable=False)] +
            make_hidden_layers(self.hidden_dim, self.hidden_layers) +
            [Dense(self.output_dim, name='out_layer')]

        )
        cells = [
            StackedRNNCells(self.output_dim),
            StackedRNNCells(self.output_dim),
            StackedRNNCells(self.output_dim),
        ]
        '''

        self.model = Sequential(
            [
                Embedding(self.embedding.shape[0], self.embedding.shape[1], weights=[self.embedding], trainable=False,
                          input_length=self.input_dim),
                RNN([SimpleRNNCell(self.hidden_dim), SimpleRNNCell(int(self.hidden_dim/2)), SimpleRNNCell(self.output_dim)]),
                Dropout(0.2)
            ])
        self.model.summary()
