from typing import Dict

import torch
from tensorflow.python.keras.optimizers import Adagrad, Adadelta, Adam
from tensorflow.python.training.adam import AdamOptimizer


def get_params(algorithm, dataset) -> Dict:
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
        return {'batch_size': 100,
                'num_epochs': 5,
                'hidden_dim': 208,
                'layer_dim': 1,
                'input_function': 'relu',
                'hidden_layers': [('rnn', None)],
                'output_function': 'selu',
                'optimizer': Adagrad(lr=0.075),
                'learning_rate': 'No'}
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
    elif algorithm == 'MLP':
        re_dict['num_epochs'] = 20
        if dsname == 'SpamHam':
            re_dict['class_weights'] = torch.FloatTensor([1, 2])
        elif dsname == 'Spamassassin':
            re_dict['class_weights'] = torch.FloatTensor([1, 100])
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
    return re_dict
