import datetime
import logging
import random

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
from Algorithms.MLPT import MLP_tensorflow
from DatasetsConsumers.Newsgroups import Newsgroups
# Load dataset
from Glove.glovemodel import GloVe
from rootfile import ROOTPATH
import os

log_file = ROOTPATH + 'Results/resultsfile.csv'

if not os.path.isfile(log_file):
    header_info = ['Avg FScore', 'Num Epochs', 'Hidden Dim', 'Layer Dim', 'Learning Rate', 'Input Layer',
                   'Hidden Layers', 'Output Layer', 'Precision', 'Recall', 'FScore', 'Timestamp']
    with open(log_file, 'w+') as f:
        f.write(','.join(header_info) + '\n')


def log_to_file(parameters, precision, recall, fscore):
    avg = sum(fscore) / len(fscore)
    log_string = "{},{},{},{},{},{},{},{},{},{},{},{}".format(
        avg,
        str(parameters['num_epochs']),
        str(parameters['hidden_dim']),
        str(parameters['layer_dim']),
        str(parameters['learning_rate']),
        str(parameters['input_layer'].name),
        str([i.name for i in parameters['hidden_layers']]),
        str(parameters['output_layer'].name),
        np.array2string(precision, separator=';', max_line_width=500),
        np.array2string(recall, separator=';', max_line_width=500),
        np.array2string(fscore, separator=';', max_line_width=500),
        datetime.datetime.now())
    with open(log_file, 'a+') as f:
        f.write(log_string + '\n')


def pick_hidden_layers(num_layers, dim):
    possible_layers = [tf.keras.layers.LeakyReLU(dim),
                       tf.keras.layers.ELU(dim),
                       tf.keras.layers.ReLU(random.randint(1, 100) / 100, random.randint(1, 100) / 100,
                                            random.randint(1, 50)),
                       #tf.keras.layers.Softmax(random.randint(-2, 2)),
                       tf.keras.layers.Dense(dim, activation=pick_activation_function())
                       ]
    return [possible_layers[random.randint(0, len(possible_layers) - 1)] for _ in range(num_layers)]


def pick_activation_function():
    possible_activations = ["relu", "softmax", "sigmoid", "elu", "selu", "softplus",
                            "softsign", "tanh"]
    return possible_activations[random.randint(0, len(possible_activations) - 1)]


counter = 1
dataset_consumer = Newsgroups()
algorithm = MLP_tensorflow

emails, labels = dataset_consumer.load(True)
glove = GloVe(200)
features = glove.get_features(emails, dataset_consumer)
print("Running algorithm:", algorithm.get_name())
while True:
    layerdim = random.randint(1, 5)
    hiddendim = random.randint(10, 500)
    parameters = {
        'batch_size': 128,
        'num_epochs': 50,
        'hidden_dim': hiddendim,
        'layer_dim': layerdim,
        'learning_rate': random.randint(1, 200) / 1000,
        'input_layer': tf.keras.layers.Dense(features.shape[1], activation=pick_activation_function()),
        'hidden_layers': pick_hidden_layers(layerdim, hiddendim),
        'output_layer': tf.keras.layers.Dense(len(set(labels)), activation=pick_activation_function()),
        # 'class_weights': None,
        'dropout': random.randint(1, 50) / 100,
        # 'max_len': 1024,
        'output_dim': len(set(labels)),
        'input_dim': features.shape[1]
    }
    print("\n#### STARTING RUN NUMBER {} #####\n".format(counter))

    data_to_plot, y_test, rounded_predictions = algorithm.run_train(dataset_consumer, features, labels,
                                                                    parameters, None, None,
                                                                    emails)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, rounded_predictions)
    print("\nPrecision: ", precision)
    print("\nRecall: ", recall)
    print("\nFscore: ", fscore)
    print("\n")
    print("Avg fScore:", (sum(fscore) / len(fscore)))

    log_to_file(parameters, precision, recall, fscore)
    counter += 1
