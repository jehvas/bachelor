import datetime
import logging

import math
import random

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
from Algorithms.MLPT import MLP_tensorflow, RNN_tensorflow, Bi_LSTM_tensorflow
from DatasetsConsumers.Newsgroups import Newsgroups
# Load dataset
from Glove.glovemodel import GloVe
from rootfile import ROOTPATH
import os

log_file = ROOTPATH + 'Results/resultsfile.csv'

if not os.path.isfile(log_file):
    header_info = ['Avg FScore', 'Num Epochs', 'Hidden Dim', 'Learning Rate', 'Input Layer',
                   'Middle Layers', 'Output Layer', 'Precision', 'Recall', 'FScore', 'Timestamp']
    with open(log_file, 'w+') as f:
        f.write(','.join(header_info) + '\n')


def log_to_file(parameters, precision, recall, fscore):
    avg = sum(fscore) / len(fscore)
    log_string = "{},{},{},{},{},{},{},{},{},{},{}".format(
        avg,
        str(parameters['num_epochs']),
        str(parameters['hidden_dim']),
        str(parameters['learning_rate']),
        str(parameters['input_function']),
        ";".join("(%s;%s)" % tup for tup in parameters['middle_layers']),
        str(parameters['output_function']),
        np.array2string(precision, separator=';', max_line_width=500),
        np.array2string(recall, separator=';', max_line_width=500),
        np.array2string(fscore, separator=';', max_line_width=500),
        datetime.datetime.now())
    with open(log_file, 'a+') as f:
        f.write(log_string + '\n')


def generate_middle_layers(num_layers):
    """
    Generate layers that are randomly filled with dropout layers.
    Returns: List of tuple (layer_type, parameter)
    Parameter is ether an activation function for the hidden layer, or a dropout percentage for the dropout layer
    """
    layers = []
    for i in range(num_layers):
        dropout_chance = int(random.randint(1, 2) / 2) * random.randint(1, 80) / 100  # 50% chance to be 0
        if dropout_chance > 0:
            layers.append(('dropout', dropout_chance))
        layers.append(('hidden', pick_random_activation_function()))
    dropout_chance = int(random.randint(1, 2) / 2) * random.randint(1, 80) / 100  # 50% chance to be 0
    if dropout_chance > 0:
        layers.append(('dropout', dropout_chance))
    return layers
    '''possible_layers = [tf.keras.layers.LeakyReLU(dim),
                       tf.keras.layers.ELU(dim),
                       tf.keras.layers.ReLU(random.randint(1, 100) / 100,
                                            random.randint(1, 100) / 100,
                                            random.randint(1, 50)),
                       # tf.keras.layers.Softmax(random.randint(-2, 2)),
                       tf.keras.layers.Dense(dim, activation=pick_activation_function())
                       ]
    return [possible_layers[random.randint(0, len(possible_layers) - 1)] for _ in range(num_layers)]'''


def pick_random_activation_function():
    possible_activations = ["relu", "softmax", "sigmoid", "elu", "selu", "softplus",
                            "softsign", "tanh"]
    return possible_activations[random.randint(0, len(possible_activations) - 1)]


counter = 1
dataset_consumer = Newsgroups()
algorithm = Bi_LSTM_tensorflow

emails, labels = dataset_consumer.load(True)
glove = GloVe(200)
features = glove.get_features(emails, dataset_consumer)
print("Running algorithm:", algorithm.get_name())
while True:
    n_hidden = 1 # 4 - int(math.log10(random.randint(10, 9000)))
    hiddendim = random.randint(10, 500)
    output_dim = len(set(labels))
    parameters = {
        'batch_size': 128,
        'num_epochs': 1,
        'hidden_dim': hiddendim,
        'learning_rate': random.randint(1, 200) / 1000,
        'input_function': pick_random_activation_function(),
        'middle_layers': generate_middle_layers(n_hidden),
        'output_function': pick_random_activation_function(),
        'output_dim': output_dim,
        'input_dim': features.shape[1],
    }
    # 'class_weights': None,
    # 'max_len': 1024,
    print("\n#### STARTING RUN NUMBER {} #####\n".format(counter))
    print(str(parameters))
    data_to_plot, y_test, rounded_predictions = algorithm.run_train(dataset_consumer, features, labels,
                                                                    parameters, emails)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, rounded_predictions)
    # print("\nPrecision: ", precision)
    # print("\nRecall: ", recall)
    # print("\nFscore: ", fscore)
    # print("\n")
    print("Avg fScore:", (sum(fscore) / len(fscore)))

    log_to_file(parameters, precision, recall, fscore)
    counter += 1
