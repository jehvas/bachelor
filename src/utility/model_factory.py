from tensorflow.python.keras.layers import LSTM, Dense, Dropout, Bidirectional, RNN, SimpleRNNCell


def make_hidden_layers(hidden_dim, middle_layers):
    layers = []
    prev_layer_type = ''
    for layer_type, param in middle_layers:
        if layer_type == prev_layer_type:  # Don't add duplicate layers
            continue
        prev_layer_type = layer_type
        if layer_type == 'hidden':
            layers.append(Dense(hidden_dim, activation=param))
        elif layer_type == 'dropout':
            layers.append(Dropout(param))
        elif layer_type == 'rnn':
            layers.append(RNN(SimpleRNNCell(hidden_dim)))
        elif layer_type == 'bi-lstm':
            layers.append(Bidirectional(LSTM(hidden_dim)))
    return layers
