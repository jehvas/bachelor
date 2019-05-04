import os
import sys
import time

import numpy as np


import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python import set_random_seed
from Algorithms import SVM, Perceptron
from Algorithms.Bi_LSTM_tensorflow import Bi_LSTM_Tensorflow
from Algorithms.MLP_tensorflow import MLP_Tensorflow
from Algorithms.RNN_tensorflow import RNN_Tensorflow
from DatasetsConsumers.EnronEvidence import EnronEvidence
from DatasetsConsumers.EnronFinancial import EnronFinancial
from DatasetsConsumers.Newsgroups import Newsgroups
from DatasetsConsumers.Spamassassin import Spamassassin
from DatasetsConsumers.Trustpilot import Trustpilot
from Glove.glovemodel import GloVe
from rootfile import ROOTPATH
from utility.Random_Parameters import get_random_params
from utility.undersample_split import resize_under_sample, under_sample_split
from utility.utility import log_to_file, setup_result_folder

algorithm_dict = {
    "all": [SVM, Perceptron, MLP_Tensorflow(), RNN_Tensorflow(), Bi_LSTM_Tensorflow()],
    "svm": [SVM],
    "perceptron": [Perceptron],
    "mlp": [MLP_Tensorflow()],
    "rnn": [RNN_Tensorflow()],
    "bi_lstm": [Bi_LSTM_Tensorflow()]
}
newsgroup = Newsgroups()
dataset_dict = {
    "all": [Newsgroups(), Spamassassin(), EnronEvidence(), EnronFinancial(), Trustpilot()],
    "newsgroups": [Newsgroups()],
    "spamassassin": [Spamassassin()],
    "enronevidence": [EnronEvidence()],
    "enronfinancial": [EnronFinancial()],
    "trustpilot": [Trustpilot()]
}

datasets_to_use = [Newsgroups()]
algorithms_to_use = [MLP_Tensorflow()]
amount = 10
# Check arguments
if len(sys.argv) != 4 or not (sys.argv[1].lower() in algorithm_dict and sys.argv[2].lower() in dataset_dict):
    print("")
    print("There was an error in the program arguments")
    print("There must be 3 arguments: an algorithm, a dataset and a count for how many times it should run")
    print("Possible algorithms:")
    for x in algorithm_dict.keys():
        print("\t" + x)
    print("Possible datasets:")
    for x in dataset_dict.keys():
        print("\t" + x)
    # exit()
else:
    algorithms_to_use = algorithm_dict[sys.argv[1].lower()]
    datasets_to_use = dataset_dict[sys.argv[2].lower()]
    amount = int(sys.argv[3])

for dataset in datasets_to_use:
    emails, labels = dataset.load(True)
    emails, labels = resize_under_sample(emails, labels)
    glove = GloVe(300)

    weights_matrix, features_from_matrix = glove.get_weights_matrix(emails, dataset)
    features_from_glove = glove.get_features(emails, dataset)

    for algorithm in algorithms_to_use:
        print("Running algorithm:", algorithm.get_name())

        if not os.path.exists(ROOTPATH + "Results/" + algorithm.get_name() + "/" + dataset.get_name() + "/plots"):
            os.makedirs(ROOTPATH + "Results/" + algorithm.get_name() + "/" + dataset.get_name() + "/plots")

        needs_weight_matrix = False
        '''(
                algorithm.get_name() == "RNN_Tensorflow"
                or algorithm.get_name() == "MLP_Tensorflow"
                #or algorithm.get_name() == "Bi_LSTM_Tensorflow"
        )'''

        setup_result_folder(algorithm.get_name(), dataset.get_name())
        best_fscore = 0
        best_fscore_list = []
        output_dim = len(set(labels))

        features = features_from_matrix if needs_weight_matrix else features_from_glove
        matrix = weights_matrix if needs_weight_matrix else None
        assert not np.any(np.isnan(features))
        # features = features[:1000]

        # Create training data
        # x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1, stratify=labels)
        x_train, x_test, y_train, y_test = under_sample_split(features, labels, test_size=0.2, random_state=1)
        # y_test = to_categorical(np.asarray(y_test))
        # y_train = to_categorical(np.asarray(y_train))
        for counter in range(1, (amount + 1)):
            np.random.seed(1)
            set_random_seed(1)
            print("#### STARTING RUN NUMBER {} #####".format(counter))

            parameters = get_random_params(algorithm.get_name(), features.shape[1], output_dim)
            print(str(parameters))

            start_time = time.time()
            # try:
            algorithm.run_train(dataset, (x_train, y_train), (x_test, y_test), parameters, embedding=matrix)

            # except Exception as e:
            #    print("Caught exception: " + str(e))
            #    continue

            avg_fscore = np.average(algorithm.fscore)
            if avg_fscore > best_fscore:
                print('New champion! {}'.format(avg_fscore))
                best_fscore = avg_fscore
                algorithm.plot_data(dataset.get_name(), counter)

            time_taken = time.time() - start_time
            # precision, recall, fscore, support = precision_recall_fscore_support(y_test, predictions)

            # avg_fscore = (sum(fscore) / len(fscore))
            # print("Avg fScore:", avg_fscore)
            file_path = ROOTPATH + "Results/" + algorithm.get_name() + "/" + dataset.get_name() + "/"
            log_to_file(parameters, algorithm.fscore, file_path + "resultsfile.csv", time_taken, algorithm.guid)
