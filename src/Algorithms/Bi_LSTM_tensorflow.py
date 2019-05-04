from Algorithms.AbstractTensorflowAlgorithm import AbstractTensorflowAlgorithm
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Embedding, Dense, LSTM, Activation, Bidirectional, CuDNNLSTM
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
            [Embedding(self.embedding.shape[0], self.embedding.shape[1],
                       input_length=self.input_dim, trainable=True),
             Bidirectional(CuDNNLSTM(128,
                                recurrent_initializer='glorot_uniform')),
             Dense(20, activation='softmax')])
        self.model.summary()
