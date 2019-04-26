import math
import random
from typing import Dict

from tensorflow.python.training.adadelta import AdadeltaOptimizer
from tensorflow.python.training.adagrad import AdagradOptimizer
from tensorflow.python.training.adagrad_da import AdagradDAOptimizer
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.ftrl import FtrlOptimizer
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from tensorflow.python.training.momentum import MomentumOptimizer
from tensorflow.python.training.proximal_adagrad import ProximalAdagradOptimizer
from tensorflow.python.training.proximal_gradient_descent import ProximalGradientDescentOptimizer
from tensorflow.python.training.rmsprop import RMSPropOptimizer


def get_random_params(algorithm, input_dim, output_dim) -> Dict:
    if algorithm == 'RNN_Tensorflow' or algorithm == 'MLP_Tensorflow' or algorithm == 'Bi-LSTM_Tensorflow':
        layer_dim = 4 - int(math.log10(random.randint(10, 9000)))
        hidden_dim = random.randint(10, 500)
        optimizer, lr = pick_optimizer()
        return {
            'hidden_dim': hidden_dim,
            'layer_dim': layer_dim,
            'input_function': pick_random_activation_function(),
            'hidden_layers': generate_middle_layers(layer_dim, algorithm),
            'output_function': pick_random_activation_function(),
            'optimizer': optimizer,
            'learning_rate': lr,
            'output_dim': output_dim,
            'input_dim': input_dim,
            'loss_function': random.choice(loss_functions),
        }

    elif algorithm == 'SVM':
        return {
            'loss_function': random.choice(["hinge", "squared_hinge"]),
            'class_weights': pick_random_class_weights(output_dim),
            'penalty': random.choice(["l2"])
        }
    elif algorithm == 'Perceptron':
        return {
            'class_weights': random.choice([pick_random_class_weights(output_dim), "balanced"]),
            'penalty': random.choice([None, "l2", "l1", "elasticnet"])
        }


def pick_random_activation_function():
    possible_activations = ["relu", "softmax", "sigmoid", "elu", "selu", "softplus",
                            "softsign", "tanh"]
    return random.choice(possible_activations)


def pick_optimizer():
    random_lr = random.randint(1, 1000) / 10000
    possible_optimizers = [AdagradOptimizer(learning_rate=random_lr),
                           GradientDescentOptimizer(learning_rate=random_lr),
                           AdadeltaOptimizer(learning_rate=random_lr),
                           AdamOptimizer(learning_rate=random_lr),
                           FtrlOptimizer(learning_rate=random_lr),
                           ProximalAdagradOptimizer(learning_rate=random_lr),
                           ProximalGradientDescentOptimizer(learning_rate=random_lr),
                           RMSPropOptimizer(learning_rate=random_lr)
                           ]
                           #MomentumOptimizer(learning_rate=random_lr)]
    optimizer_to_return = random.choice(possible_optimizers)
    return optimizer_to_return, str(random_lr)


def pick_random_class_weights(num_labels):
    classes = [i for i in range(num_labels)]
    class_weight_dic = {}
    for i in classes:
        class_weight_dic[i] = random.randint(1, 100)
    return class_weight_dic


loss_functions = [# 'mean_squared_error',
                  'mean_absolute_error',
                  'mean_absolute_percentage_error',
                  'mean_squared_logarithmic_error',
                  'squared_hinge',
                  'hinge',
                  'categorical_hinge',
                  'logcosh',
                  # 'categorical_crossentropy',
                  'sparse_categorical_crossentropy',
                  # 'binary_crossentropy',
                  'kullback_leibler_divergence',
                  'poisson',
                  'cosine_proximity'
                  ]


def generate_middle_layers(num_layers, algorithm):
    """
    Generate layers that are randomly filled with dropout layers.
    Returns: List of tuple (layer_type, parameter)
    Parameter is ether an activation function for the hidden layer, or a dropout percentage for the dropout layer
    """
    layers = []
    # Special networks must have their corresponding specific layer.
    if algorithm == "RNN_Tensorflow":
        layers.append(('rnn', ""))
    elif algorithm == "Bi-LSTM_Tensorflow":
        layers.append(("bi-lstm", ""))

    for i in range(num_layers - len(layers)):
        dropout_chance = int(random.randint(1, 2) / 2) * random.randint(1, 80) / 100  # 50% chance to be 0
        if dropout_chance > 0:
            layers.append(('dropout', dropout_chance))
        else:
            layers.append(('hidden', pick_random_activation_function()))
    random.shuffle(layers)
    dropout_chance = int(random.randint(1, 2) / 2) * random.randint(1, 80) / 100  # 50% chance to be 0
    if dropout_chance > 0:
        layers.append(('dropout', dropout_chance))
    return layers