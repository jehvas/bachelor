import tensorflow as tf

from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, GlobalAveragePooling1D, LeakyReLU

from Algorithms.AbstractTensorflowAlgorithm import AbstractTensorflowAlgorithm
from utility.model_factory import make_hidden_layers


class MLP_Tensorflow(AbstractTensorflowAlgorithm):
    def generate_model(self):
        self.model = Sequential(
            [
                Embedding(self.embedding.shape[0], self.embedding.shape[1], trainable=True, weights=[self.embedding],
                          input_length=self.input_dim),
                GlobalAveragePooling1D(),
                Dense(self.hidden_dim, input_dim=self.input_dim)] +
            # make_hidden_layers(self.hidden_dim, self.hidden_layers) +
            [
                Dense(self.output_dim, name='out_layer', activation='softmax'),
            ]
        )
        self.model.summary()
