from Algorithms.AbstractTensorflowAlgorithm import AbstractTensorflowAlgorithm
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Embedding, Dense, LSTM, Activation, Bidirectional, Dropout
from utility.model_factory import make_hidden_layers
import tensorflow as tf

class Bi_LSTM_Tensorflow(AbstractTensorflowAlgorithm):
    def generate_model(self, middle_layers, input_shape):
        # if tf.test.gpu_device_name():

        '''
        ("Bi_LSTM", random.randint(10, 300), "linear"),
                                       ("LeakyReLU", "", ""),
                                       ("Dropout", random.randint(0, 5) / 10, ""),
                                       ("Bi_LSTM", random.randint(10, 300), "linear"),
                                       ("LeakyReLU", "", ""),
                                       ("Dropout", random.randint(0, 5) / 10, ""),
                                       ("Dense", input_dim, "linear"),
                                       ("LeakyReLU", "", ""),
                                       ("Dropout", random.randint(0, 5) / 10, ""),
                                       ("Dense", output_dim, 'softmax')
        '''
        self.model = Sequential(make_hidden_layers(middle_layers, input_shape))
