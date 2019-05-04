from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Embedding, Dense, RNN, SimpleRNNCell, Dropout, LeakyReLU, Softmax
from tensorflow.python.ops.gen_nn_ops import LogSoftmax, leaky_relu

from Algorithms.AbstractTensorflowAlgorithm import AbstractTensorflowAlgorithm


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
                Embedding(self.embedding.shape[0], self.embedding.shape[1], trainable=False, weights=[self.embedding],
                          input_length=self.input_dim),
                RNN(SimpleRNNCell(128, activation='tanh')),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.1),
                # return_sequences=True,
                # stateful=True
                # SimpleRNNCell(int(self.hidden_dim/2)),
                # SimpleRNNCell(self.output_dim)
                # , stateful=True),
                # Softmax(),
                # LeakyReLU(),
                Dense(self.output_dim, name='out_layer', activation='softmax'),
            ])
        self.model.summary()
