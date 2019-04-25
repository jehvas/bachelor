from tensorflow.python.keras import Sequential, Input
from tensorflow.python.keras.layers import LSTM, Dense, Dropout, Bidirectional, RNN, SimpleRNNCell, Embedding, Flatten


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
    return layers


def generate_mlp_model(input_dim, hidden_dim, middle_layers, output_dim, input_function, output_function):
    return Sequential(
        [Dense(input_dim, input_dim=input_dim, activation=input_function)] +
        make_hidden_layers(hidden_dim, middle_layers) +
        [Dense(output_dim, activation=output_function)]
    )


def generate_bi_lstm_model(input_dim, hidden_dim, middle_layers, output_dim, input_function, output_function,
                           embedding):
    return Sequential(
        [Input(shape=[256]), Embedding(embedding.shape[0], embedding.shape[1], weights=[embedding], input_length=256)] +
        make_hidden_layers(hidden_dim, middle_layers) +
        [Dense(output_dim, activation=output_function)]
    )


def generate_rnn_model(input_dim, hidden_dim, middle_layers, output_dim, input_function, output_function,
                       embedding):
    return Sequential(
        [Embedding(embedding.shape[0], embedding.shape[1], weights=[embedding]),
         Dense(input_dim, input_dim=input_dim, activation=input_function)] +
        make_hidden_layers(hidden_dim, middle_layers) +
        [Dense(output_dim, name='out_layer', activation=output_function)]
    )
