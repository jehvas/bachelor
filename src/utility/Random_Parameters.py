import random
import tensorflow as tf
from typing import Dict

from keras.optimizers import Adam, Adamax, Nadam, SGD, Adagrad, RMSprop, Adadelta


def get_random_params(algorithm, input_dim, output_dim) -> Dict:
    if algorithm == 'RNN_tensorflow' or algorithm == 'MLP_Tensorflow' or algorithm == 'Bi_LSTM_tensorflow':
        layer_dim = random.randint(1, 1)
        hidden_dim = random.randint(10, 500)
        optimizer, lr = pick_optimizer()
        return {
            'batch_size': 128,
            'num_epochs': 50,
            'hidden_dim': hidden_dim,
            'layer_dim': layer_dim,
            'input_function': pick_activation_function(),
            'hidden_layers': pick_hidden_layers(layer_dim, hidden_dim),
            'output_function': pick_activation_function(),
            'optimizer': optimizer,
            'learning_rate': lr,
            # 'class_weights': None,
            'dropout': random.randint(1, 80) / 100,
            # 'max_len': 1024,
            'output_dim': output_dim,
            'input_dim': input_dim,
            'use_dropout': True if random.randint(1, 2) == 1 else False,
        }

    elif algorithm == 'SVM':
        return {
            'loss': ["hinge", "squared_hinge"][random.randint(0, 1)],
            'class_weights': {0: 1, 1: 2}
        }
    elif algorithm == 'Perceptron':
        return {
            'alpha': random.randint(1, 100) / 1000
        }


def pick_hidden_layers(num_layers, dim):
    possible_layers = [tf.keras.layers.LeakyReLU(dim),
                       tf.keras.layers.ELU(dim),
                       tf.keras.layers.ReLU(random.randint(1, 100) / 100,
                                            random.randint(1, 100) / 100,
                                            random.randint(1, 50)),
                       # tf.keras.layers.Softmax(random.randint(-2, 2)),
                       tf.keras.layers.Dense(dim, activation=pick_activation_function())
                       ]
    return [possible_layers[random.randint(0, len(possible_layers) - 1)] for _ in range(num_layers)]


def pick_activation_function():
    possible_activations = ["relu", "softmax", "sigmoid", "elu", "selu", "softplus",
                            "softsign", "tanh"]
    return possible_activations[random.randint(0, len(possible_activations) - 1)]


def pick_optimizer():
    random_lr = random.randint(1, 200) / 1000
    possible_optimizers = [Adam(lr=random_lr), RMSprop(lr=random_lr), Adadelta(), Adagrad(lr=random_lr),
                           Adamax(lr=random_lr), Nadam(lr=random_lr), SGD(lr=random_lr)]
    optimizer_to_return = possible_optimizers[random.randint(0, len(possible_optimizers)-1)]
    if "Adadelta" in optimizer_to_return.lr.name:
        return optimizer_to_return, "None"
    return optimizer_to_return, str(random_lr)
