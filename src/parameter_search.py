import os
import sys
import time
from collections import Counter

import gc
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python import set_random_seed, reset_default_graph, ops
from tensorflow.python.keras.backend import clear_session

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
dataset_modes = [
    "standard",
    "2000",
    "equal"
]

datasets_to_use = [Newsgroups()]
algorithms_to_use = [MLP_Tensorflow()]
amount = 10
dataset_mode = 0
# Check arguments
if len(sys.argv) != 5 or not (sys.argv[1].lower() in algorithm_dict and sys.argv[2].lower() in dataset_dict and sys.argv[4].lower() in dataset_modes):
    print("")
    print("There was an error in the program arguments")
    print("There must be 3 arguments: an algorithm, a dataset and a count for how many times it should run")
    print("Possible algorithms:")
    for x in algorithm_dict.keys():
        print("\t" + x)
    print("Possible datasets:")
    for x in dataset_dict.keys():
        print("\t" + x)
    print("Possible dataset modes:")
    for i, x in enumerate(dataset_modes):
        print("\t" + x)
    # exit()
else:
    algorithms_to_use = algorithm_dict[sys.argv[1].lower()]
    datasets_to_use = dataset_dict[sys.argv[2].lower()]
    amount = int(sys.argv[3])
    dataset_mode = sys.argv[4]
    if not os.path.exists(ROOTPATH + "output/" + dataset_mode):
        os.makedirs(ROOTPATH + "output/" + dataset_mode)

for dataset in datasets_to_use:
    emails, labels = dataset.load(True, dataset_mode)

    if dataset_mode == "2000":
        emails, labels = resize_under_sample(emails, labels)

    glove = GloVe(300)

    # weights_matrix, features_from_matrix = glove.get_weights_matrix(emails, dataset, dataset_mode)
    features_from_glove = glove.get_features(emails, dataset, dataset_mode)

    for algorithm in algorithms_to_use:
        print("Running algorithm:", algorithm.get_name())

        if not os.path.exists(ROOTPATH + "Results/" + dataset_mode + "/" + algorithm.get_name() + "/" + dataset.get_name() + "/plots"):
            os.makedirs(ROOTPATH + "Results/" + dataset_mode + "/" + algorithm.get_name() + "/" + dataset.get_name() + "/plots")

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

        # features = features_from_matrix if needs_weight_matrix else features_from_glove
        # matrix = weights_matrix if needs_weight_matrix else None
        features = features_from_glove
        matrix = None
        assert not np.any(np.isnan(features))

        # Create training data
        if dataset_mode != "equal":
            x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1, stratify=labels)
        else:
            x_train, x_test, y_train, y_test = under_sample_split(features, labels, test_size=0.2, random_state=1)
        print(Counter(y_train))
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
            algorithm.run_train(dataset, (x_train, y_train), (x_test, y_test), parameters)

            # except Exception as e:
            #    print("Caught exception: " + str(e))
            #    continue

            avg_fscore = np.average(algorithm.fscore)
            if avg_fscore > best_fscore:
                print('New champion! {}'.format(avg_fscore))
                best_fscore = avg_fscore
                algorithm.plot_data(dataset.get_name(), counter, dataset_mode)

            time_taken = time.time() - start_time

            file_path = ROOTPATH + "Results/" + dataset_mode + "/" + algorithm.get_name() + "/" + dataset.get_name() + "/"
            log_to_file(parameters, algorithm.fscore, file_path + "resultsfile.csv", time_taken, algorithm.guid)
            clear_session()
            reset_default_graph()
            ops.reset_default_graph()
            gc.collect()
