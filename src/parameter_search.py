import datetime
import logging
import random

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from Algorithms.MLPT import MLP_tensorflow
from DatasetsConsumers.Newsgroups import Newsgroups
# Load dataset
from Glove.glovemodel import GloVe
from rootfile import ROOTPATH
import os

log_file = ROOTPATH + 'Results/resultsfile.csv'

if not os.path.isfile(log_file):
    header_info = ['Avg FScore', 'Num Epochs', 'Hidden Dim', 'Layer Dim', 'Learning Rate', 'Dropout',
                   'Precision', 'Recall', 'FScore', 'Timestamp']
    with open(log_file, 'w+') as f:
        f.write(','.join(header_info) + '\n')


def log_to_file(parameters, precision, recall, fscore):
    avg = sum(fscore) / len(fscore)
    log_string = "{},{},{},{},{},{},{},{},{},{}".format(
        avg,
        str(parameters['num_epochs']),
        str(parameters['hidden_dim']),
        str(parameters['layer_dim']),
        str(parameters['learning_rate']),
        str(parameters['dropout']),
        np.array2string(precision, separator=';', max_line_width=500),
        np.array2string(recall, separator=';', max_line_width=500),
        np.array2string(fscore, separator=';', max_line_width=500),
        datetime.datetime.now())
    with open(log_file, 'a+') as f:
        f.write(log_string + '\n')


counter = 1
dataset_consumer = Newsgroups()
algorithm = MLP_tensorflow

emails, labels = dataset_consumer.load(True)
glove = GloVe(200)
features = glove.get_features(emails, dataset_consumer)
print("Running algorithm:", algorithm.get_name())
while True:
    parameters = {
        'batch_size': 128,
        'num_epochs': 50,
        'hidden_dim': random.randint(10, 500),
        'layer_dim': 1,
        'learning_rate': random.randint(1, 200) / 1000,
        # 'class_weights': None,
        'dropout': random.randint(1, 99) / 100,
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
