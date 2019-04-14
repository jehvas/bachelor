import tensorflow as tf
from tensorflow.python.keras.layers import LSTM, Dense, Dropout, Bidirectional, RNN, SimpleRNNCell


def make_hidden_layers(hidden_dim, middle_layers):
    layers = []
    for layer_type, param in middle_layers:
        if layer_type == 'hidden':
            layers.append(Dense(hidden_dim, activation=param))
        elif layer_type == 'dropout':
            layers.append(Dropout(param))
    return layers


def generate_model(input_dim, hidden_dim, middle_layers, output_dim, input_function, output_function,
                   isLSTM=False, isRNN=False):
    return tf.keras.Sequential(
        [Dense(input_dim, activation=input_function)] + \
        ([Bidirectional(LSTM(hidden_dim))] if isLSTM else []) + \
        ([RNN(SimpleRNNCell(hidden_dim))] if isRNN else []) + \
        make_hidden_layers(hidden_dim, middle_layers) + \
        [Dense(output_dim, activation=output_function)]
    )
