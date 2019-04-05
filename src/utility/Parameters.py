from typing import Dict

import torch


def get_params(algorithm, dataset) -> Dict:
    dsname = type(dataset).__name__
    re_dict = {
        'batch_size': 1000,
        'num_epochs': 50,
        'hidden_dim': 128,
        'layer_dim': 1,
        'learning_rate': 0.01,
        'class_weights': None,
    }
    if algorithm == 'RNN':
        pass
    elif algorithm == 'MLP_tensorflow':
        pass
    elif algorithm == 'MLP':
        if dsname == 'SpamHam':
            re_dict['class_weights'] = torch.FloatTensor([1, 2])
        elif dsname == 'Spamassassin':
            re_dict['class_weights'] = torch.FloatTensor([1, 100])
    elif algorithm == 'SVM':
        if dsname == 'SpamHam':
            re_dict['class_weights'] = {0: 1, 1: 2}
        elif dsname == 'Spamassassin':
            re_dict['class_weights'] = {0: 1, 1: 2}
    return re_dict
