import os
import sys
import time

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from Algorithms import SVM, Perceptron, RNN_tensorflow, MLP_tensorflow, Bi_LSTM_tensorflow
from DatasetsConsumers.EnronEvidence import EnronEvidence
from DatasetsConsumers.EnronFinancial import EnronFinancial
from DatasetsConsumers.Newsgroups import Newsgroups
from DatasetsConsumers.Spamassassin import Spamassassin
from DatasetsConsumers.Trustpilot import Trustpilot
from Glove.glovemodel import GloVe
from rootfile import ROOTPATH
from utility.Random_Parameters import get_random_params
from utility.confusmatrix import plot_confusion_matrix
from utility.utility import log_to_file, setup_result_folder

algorithm_dict = {
    "all": [SVM, Perceptron, MLP_tensorflow, RNN_tensorflow, Bi_LSTM_tensorflow],
    "svm": [SVM],
    "perceptron": [Perceptron],
    "mlp": [MLP_tensorflow],
    "rnn": [RNN_tensorflow],
    "bi-lstm": [Bi_LSTM_tensorflow]
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
algorithms_to_use = [RNN_tensorflow]
amount = 99999
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
    glove = GloVe(50)

    weights_matrix, features_from_matrix = glove.get_weights_matrix(emails, dataset)
    features_from_glove = glove.get_features(emails, dataset)

    for algorithm in algorithms_to_use:
        print("Running algorithm:", algorithm.get_name())

        needs_weight_matrix = (algorithm.get_name() == "RNN_Tensorflow" or
                               algorithm.get_name() == "Bi-LSTM_Tensorflow")

        setup_result_folder(algorithm.get_name(), dataset.get_name())
        best_fscore = 0
        output_dim = len(set(labels))

        features = features_from_matrix if needs_weight_matrix else features_from_glove
        matrix = weights_matrix if needs_weight_matrix else None
        assert not np.any(np.isnan(features))

        for counter in range(1, amount):
            print("\n#### STARTING RUN NUMBER {} #####\n".format(counter))

            parameters = get_random_params(algorithm.get_name(), features.shape[1], output_dim)
            print(str(parameters))

            start_time = time.time()

            data_to_plot, y_test, predictions = algorithm.run_train(dataset, features, labels, parameters, embedding=matrix)

            time_taken = time.time() - start_time
            precision, recall, fscore, support = precision_recall_fscore_support(y_test, predictions)

            avg_fscore = (sum(fscore) / len(fscore))
            print("Avg fScore:", avg_fscore)
            file_path = ROOTPATH + "Results/" + algorithm.get_name() + "/" + dataset.get_name() + "/"
            log_to_file(parameters, fscore, file_path + "resultsfile.csv", time_taken)

            if avg_fscore >= best_fscore:
                best_fscore = avg_fscore
                if not os.path.exists(
                        ROOTPATH + "Results/" + algorithm.get_name() + "/" + dataset.get_name() + "/plots"):
                    os.mkdir(ROOTPATH + "Results/" + algorithm.get_name() + "/" + dataset.get_name() + "/plots")
                # if len(data_to_plot) != 0:
                #    plot_data(data_to_plot[0], file_path + "/plots/" + str(counter) + "_plot_val_acc_.png")
                #    plot_data(data_to_plot[1], file_path + "/plots/" + str(counter) + "_plot_val_loss_.png")
                plot_confusion_matrix(y_test, predictions, dataset, algorithm, normalize=True,
                                      save_path=file_path + "/plots/" + str(counter) + "_confusmatrix_.png")
