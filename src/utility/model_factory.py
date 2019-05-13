import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Bidirectional, RNN, SimpleRNNCell, CuDNNLSTM, Dropout, LeakyReLU


def make_hidden_layers(middle_layers, input_shape):
    layers = []
    for idx, layer_info in enumerate(middle_layers):
        layer_type, size, activation_func = layer_info
        if layer_type == 'Dense':
            if idx == 0:
                layers.append(Dense(size, activation=activation_func, input_shape=input_shape))
            else:
                layers.append(Dense(size, activation=activation_func))
        elif layer_type == 'LeakyReLU':
            layers.append(LeakyReLU())
        elif layer_type == 'Dropout':
            layers.append(Dropout(size))
        elif layer_type == 'RNN':
            if idx == 0:
                layers.append(RNN(SimpleRNNCell(size, activation=activation_func), input_shape=input_shape,
                                  return_sequences=True))
            else:
                layers.append(RNN(SimpleRNNCell(size, activation=activation_func)))
        elif layer_type == 'Bi_LSTM':
            if tf.test.gpu_device_name():
                print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
                if idx == 0:
                    layers.append(Bidirectional(CuDNNLSTM(size, input_shape=input_shape, return_sequences=True)))
                else:
                    layers.append(Bidirectional(CuDNNLSTM(size)))
            else:
                raise Exception("This device does not support the GPU version of Tensorflow.\nPlease install needed "
                                "drivers or run on Google Colab")

    return layers
