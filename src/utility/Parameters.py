from typing import Dict

import torch


def get_params(algorithm, dataset) -> Dict:
    dsname = type(dataset).__name__
    re_dict = {
        'batch_size': 128,
        'num_epochs': 50,
        'hidden_dim': 128,
        'layer_dim': 1,
        'learning_rate': 0.01,
        'class_weights': None,
        'dropout': 0.5,
        'max_len': 128
    }
    if algorithm == 'RNN':
        pass
    elif algorithm == 'MLP_Tensorflow':
        re_dict['num_epochs'] = 1
        pass
    elif algorithm == 'MLP':
        re_dict['num_epochs'] = 1
        if dsname == 'SpamHam':
            re_dict['class_weights'] = torch.FloatTensor([1, 2])
        elif dsname == 'Spamassassin':
            re_dict['class_weights'] = torch.FloatTensor([1, 100])
    elif algorithm == 'SVM':
        if dsname == 'SpamHam':
            re_dict['class_weights'] = {0: 1, 1: 2}
        elif dsname == 'Spamassassin':
            re_dict['class_weights'] = {0: 1, 1: 2}
    elif algorithm == 'Bi-LSTM_Tensorflow':
        re_dict['num_epochs'] = 1
    elif algorithm == 'RNN_Tensorflow':
        re_dict['num_epochs'] = 1
    return re_dict
