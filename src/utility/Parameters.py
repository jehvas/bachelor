from typing import Dict

from tensorflow.python.keras.optimizers import Adagrad
from tensorflow.python.training.adadelta import AdadeltaOptimizer
from tensorflow.python.training.adam import AdamOptimizer


def get_params(algorithm, dataset):
    dsname = type(dataset).__name__
    re_dict = {
        'batch_size': 128,
        'num_epochs': 50,
        'hidden_dim': 128,
        'layer_dim': 1,
        'learning_rate': 0.01,
        'class_weights': None,
        'dropout': 0,
        'max_len': 1024

    }
    if algorithm == 'RNN_Tensorflow':
        return {'hidden_dim': 120, 'layer_dim': 2, 'input_function': 'relu',
                'hidden_layers': [('hidden', 'tanh'), ('rnn', ''), ('dropout', 0.63)], 'output_function': 'sigmoid',
                'optimizer': AdadeltaOptimizer(learning_rate=0.0757), 'learning_rate': '0.0757', 'output_dim': 20,
                'input_dim': 256}
    elif algorithm == 'MLP_Tensorflow':
        return {'batch_size': 780,
                'num_epochs': 50,
                'hidden_dim': 259,
                'layer_dim': 1,
                'input_function': 'relu',
                'hidden_layers': [('dropout', 0.23), ('hidden', 'sigmoid')],
                'output_function': 'relu',
                'optimizer': AdamOptimizer(learning_rate=0.001),
                'learning_rate': '0.001'}
    elif algorithm == 'SVM':
        if dsname == 'SpamHam':
            re_dict['class_weights'] = {0: 1, 1: 2}
            re_dict['loss_function'] = "squared_hinge"
            re_dict['penalty'] = "l2"
        elif dsname == 'Spamassassin':
            re_dict['class_weights'] = {0: 1, 1: 2}
            re_dict['loss_function'] = "squared_hinge"
            re_dict['penalty'] = "l2"
    elif algorithm == 'Bi_LSTM_Tensorflow':
        return {'batch_size': 100,
                'num_epochs': 5,
                'hidden_dim': 208,
                'layer_dim': 1,
                'input_function': 'relu',
                'hidden_layers': [('bi_lstm', None)],
                'output_function': 'selu',
                'optimizer': Adagrad(lr=0.075),
                'learning_rate': 'No'}
    elif algorithm == 'RNN_Tensorflow':
        re_dict['num_epochs'] = 1
    elif algorithm == 'Perceptron':
        re_dict['class_weights'] = {0: 1, 1: 2}
        re_dict['loss_function'] = "squared_hinge"
        re_dict['penalty'] = "l2"
    return re_dict
