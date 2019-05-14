import os

from Algorithms.Bi_LSTM_tensorflow import Bi_LSTM_Tensorflow
from Algorithms.MLP_tensorflow import MLP_Tensorflow
from Algorithms.Perceptron import Perceptron
from Algorithms.RNN_tensorflow import RNN_Tensorflow
from Algorithms.SVM import SVM
from DatasetsConsumers.EnronEvidence import EnronEvidence
from DatasetsConsumers.EnronFinancial import EnronFinancial
from DatasetsConsumers.Newsgroups import Newsgroups
from DatasetsConsumers.Spamassassin import Spamassassin
from DatasetsConsumers.Trustpilot import Trustpilot
from rootfile import ROOTPATH


def parse_arguments(argv):
    algorithm_dict = {
        "all": [SVM(), Perceptron(), MLP_Tensorflow(), RNN_Tensorflow(), Bi_LSTM_Tensorflow()],
        "SVM()": [SVM()],
        "Perceptron()": [Perceptron()],
        "mlp": [MLP_Tensorflow()],
        "rnn": [RNN_Tensorflow()],
        "bi_lstm": [Bi_LSTM_Tensorflow()]
    }
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
    if len(argv) != 5 or not (argv[1].lower() in algorithm_dict and argv[2].lower() in dataset_dict and argv[4].lower() in dataset_modes):
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
        exit()
    else:
        algorithms_to_use = algorithm_dict[argv[1].lower()]
        datasets_to_use = dataset_dict[argv[2].lower()]
        amount = int(argv[3])
        dataset_mode = argv[4]
        if not os.path.exists(ROOTPATH + "output/" + dataset_mode):
            os.makedirs(ROOTPATH + "output/" + dataset_mode)
    return algorithms_to_use, datasets_to_use, amount, dataset_mode
