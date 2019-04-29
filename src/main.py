import sys

import numpy as np
import time
from sklearn.metrics import precision_recall_fscore_support
from Algorithms import SVM, Perceptron, MLP_tensorflow, Bi_LSTM_tensorflow
from Algorithms.AbstractTensorflowAlgorithm import AbstractTensorflowAlgorithm
from Algorithms.RNN_tensorflow import RNN_Tensorflow
from Algorithms.MLP_tensorflow import MLP_Tensorflow
from Algorithms.Bi_LSTM_tensorflow import Bi_LSTM_Tensorflow
from DatasetsConsumers.EnronEvidence import EnronEvidence
from DatasetsConsumers.EnronFinancial import EnronFinancial
from DatasetsConsumers.Newsgroups import Newsgroups
from DatasetsConsumers.Spamassassin import Spamassassin
from DatasetsConsumers.Trustpilot import Trustpilot
from Glove.glovemodel import GloVe
from utility.Parameters import get_params
from utility.confusmatrix import plot_confusion_matrix

algorithms = {
    "all": [SVM, Perceptron, MLP_Tensorflow(), RNN_Tensorflow(), Bi_LSTM_Tensorflow()],
    "svm": [SVM],
    "perceptron": [Perceptron],
    "mlp": [MLP_Tensorflow()],
    "rnn": [RNN_Tensorflow()],
    "bi-lstm": [Bi_LSTM_Tensorflow()]
}
newsgroup = Newsgroups()
datasets = {
    "all": [Newsgroups(), Spamassassin(), EnronEvidence(), EnronFinancial(), Trustpilot()],
    "newsgroups": [Newsgroups()],
    "spamassassin": [Spamassassin()],
    "enronevidence": [EnronEvidence()],
    "enronfinancial": [EnronFinancial()],
    "trustpilot": [Trustpilot()]
}

datasets_to_use = [Trustpilot(), EnronEvidence(), EnronFinancial(), Newsgroups(), Spamassassin()]
algorithms_to_use = [RNN_Tensorflow()]
# Check arguments
if len(sys.argv) != 3 or not (sys.argv[1].lower() in algorithms and sys.argv[2].lower() in datasets):
    print("")
    print("There was an error in the program arguments")
    print("There must be 2 arguments: an algorithm and a dataset.")
    print("Possible algorithms:")
    for x in algorithms.keys():
        print("\t" + x)
    print("Possible datasets:")
    for x in datasets.keys():
        print("\t" + x)
    # exit()
else:
    algorithms_to_use = algorithms[sys.argv[1].lower()]
    datasets_to_use = datasets[sys.argv[2].lower()]

for dataset in datasets_to_use:
    emails, labels = dataset.load(True)
    glove = GloVe(50)

    weights_matrix, features_from_matrix = glove.get_weights_matrix(emails, dataset)
    features_from_glove = glove.get_features(emails, dataset)

    for algorithm in algorithms_to_use:
        print("Running algorithm:", algorithm.get_name())

        parameters = get_params(algorithm.get_name(), dataset)
        print(str(parameters))
        needs_weight_matrix = (algorithm.get_name() == "RNN_Tensorflow" or
                               algorithm.get_name() == "Bi-LSTM_Tensorflow")

        features = features_from_matrix if needs_weight_matrix else features_from_glove
        matrix = weights_matrix if needs_weight_matrix else None
        assert not np.any(np.isnan(features))

        parameters['output_dim'] = len(set(labels))
        parameters['input_dim'] = features.shape[1]
        start_time = time.time()
        data_to_plot, y_test, predictions = algorithm.run_train(dataset, features, labels, parameters, embedding=matrix)
        time_taken = time.time() - start_time
        print("Finished in {:.3f}".format(time_taken))
        # for plotClass in data_to_plot:
        #    plot_data(plotClass, True)

        precision, recall, fscore, support = precision_recall_fscore_support(y_test, predictions)
        print("\nPrecision: ", precision)
        print("\nRecall: ", recall)
        print("\nFscore: ", fscore)
        print("\n")
        print("Avg fScore:", (sum(fscore) / len(fscore)))

        plot_confusion_matrix(y_test, predictions, dataset, algorithm.get_name(), normalize=True)
