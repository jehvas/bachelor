import time

import datetime

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from Algorithms.MLPT import MLP_tensorflow
from DatasetsConsumers.Newsgroups import Newsgroups
# Load dataset
from Glove.glovemodel import GloVe
from rootfile import ROOTPATH
from utility.Parameters import get_params


def log_to_file(idx, parameters, precision, recall, fscore):
    avg = sum(fscore) / len(fscore)
    log_string = "{}\t{}\t{}\t{}\t{}\t{}\t{},".format(idx,
                                                      avg,
                                                      parameters,
                                                      np.array2string(precision, separator=';', max_line_width=500),
                                                      np.array2string(recall, separator=';', max_line_width=500),
                                                      np.array2string(fscore, separator=';', max_line_width=500),
                                                      datetime.datetime.now())
    with open(ROOTPATH + 'Results/resultsfile.csv', 'a+') as f:
        f.write(log_string + '\n')


counter = 0
dataset_consumer = Newsgroups()
algorithm = MLP_tensorflow

emails, labels = dataset_consumer.load(True)
glove = GloVe(200)
features = glove.get_features(emails, dataset_consumer)
print("Running algorithm:", algorithm.get_name())
while True:
    parameters = {
        'batch_size': 128,
        'num_epochs': 1,
        'hidden_dim': 128,
        'layer_dim': 1,
        'learning_rate': 0.01,
        'class_weights': None,
        'dropout': 0.5,
        'max_len': 1024,
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

    log_to_file(counter, str(parameters), precision, recall, fscore)
    counter += 1
