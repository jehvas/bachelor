from tensorflow.python.keras import Sequential, Input
from tensorflow.python.keras.layers import LSTM, Dense, Dropout, Bidirectional, RNN, SimpleRNNCell, Embedding, Flatten
import tensorflow as tf


def make_hidden_layers(hidden_dim, middle_layers):
    layers = []
    for layer_type, param in middle_layers:
        if layer_type == 'hidden':
            layers.append(Dense(hidden_dim, activation=param))
        elif layer_type == 'dropout':
            layers.append(Dropout(param))
        elif layer_type == 'rnn':
            layers.append(RNN(SimpleRNNCell(hidden_dim)))
        elif layer_type == 'bi-lstm':
            layers.append(Bidirectional(LSTM(hidden_dim)))
    for i in range(1, len(layers)):
        if type(layers[i - 1]) is type(layers[i]):
            print('IDENTICAL LAYERS!', layers)
    return layers