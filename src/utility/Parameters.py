from typing import Dict

import torch
from tensorflow.python.keras.optimizers import Adagrad


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
        return {'batch_size': 780, 'num_epochs': 50, 'hidden_dim': 259, 'layer_dim': 1, 'input_function': 'selu', 'hidden_layers': [('dropout', 0.23), ('hidden', 'relu')], 'output_function': 'relu', 'optimizer': Adagrad(lr=0.0165), 'learning_rate': '0.0165', 'dropout': 0.11, 'output_dim': 2, 'input_dim': 256, 'use_dropout': True, 'loss_function': 'squared_hinge'}
    elif algorithm == 'MLP_Tensorflow':
        pass
    elif algorithm == 'MLP':
        re_dict['num_epochs'] = 20
        if dsname == 'SpamHam':
            re_dict['class_weights'] = torch.FloatTensor([1, 2])
        elif dsname == 'Spamassassin':
            re_dict['class_weights'] = torch.FloatTensor([1, 100])
    elif algorithm == 'SVM':
        if dsname == 'SpamHam':
            re_dict['class_weights'] = {0: 1, 1: 2}
            re_dict['loss'] = "squared_hinge"
            re_dict['penalty'] = "l2"
        elif dsname == 'Spamassassin':
            re_dict['class_weights'] = {0: 1, 1: 2}
            re_dict['loss'] = "squared_hinge"
            re_dict['penalty'] = "l2"
    elif algorithm == 'Bi-LSTM_Tensorflow':
        re_dict['num_epochs'] = 10
    elif algorithm == 'RNN_Tensorflow':
        re_dict['num_epochs'] = 1
    return re_dict
