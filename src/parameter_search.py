import datetime
import logging
import random
import time

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
from Algorithms.MLPT import MLP_tensorflow
from Algorithms.Perceptron import Perceptron
from Algorithms.SVM import SVM
from DatasetsConsumers.Newsgroups import Newsgroups
# Load dataset
from Glove.glovemodel import GloVe
from rootfile import ROOTPATH
import os

from utility import Random_Parameters, utility


def pick_hidden_layers(num_layers, dim):
    possible_layers = [tf.keras.layers.LeakyReLU(dim),
                       tf.keras.layers.ELU(dim),
                       tf.keras.layers.ReLU(random.randint(1, 100) / 100,
                                            random.randint(1, 100) / 100,
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
    parameters = Random_Parameters.get_random_params(algorithm.get_name(), features.shape[1], len(set(labels)))

    print("\n#### STARTING RUN NUMBER {} #####\n".format(counter))
    start_time = time.time()
    data_to_plot, y_test, rounded_predictions = algorithm.run_train(dataset_consumer, features, labels,
                                                                    parameters, None, None,
                                                                    emails)
    end_time = time.time()
    time_taken = end_time - start_time
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, rounded_predictions)
    # print("\nPrecision: ", precision)
    # print("\nRecall: ", recall)
    # print("\nFscore: ", fscore)
    # print("\n")
    print("Avg fScore:", (sum(fscore) / len(fscore)))
    log_file = ROOTPATH + 'Results/' + algorithm.get_name() + '_resultsfile.csv'
    utility.log_to_file(parameters, precision, recall, fscore, log_file, time_taken)
    counter += 1
