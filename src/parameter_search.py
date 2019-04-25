import os
import sys
import time

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

algorithms = {
    "all": [SVM, Perceptron, MLP_tensorflow, RNN_tensorflow, Bi_LSTM_tensorflow],
    "svm": [SVM],
    "perceptron": [Perceptron],
    "mlp": [MLP_tensorflow],
    "rnn": [RNN_tensorflow],
    "bi-lstm": [Bi_LSTM_tensorflow]
}
newsgroup = Newsgroups()
datasets = {
    "all": [Newsgroups(), Spamassassin(), EnronEvidence(), EnronFinancial(), Trustpilot()],
    "newgroups": [Newsgroups()],
    "spamassassin": [Spamassassin()],
    "enronevidence": [EnronEvidence()],
    "enronfinancial": [EnronFinancial()],
    "trustpilot": [Trustpilot()]
}

datasets_to_use = [Spamassassin()]
algorithms_to_use = [Bi_LSTM_tensorflow]
amount = 99999
# Check arguments
if len(sys.argv) != 4 or not (sys.argv[1].lower() in algorithms and sys.argv[2].lower() in datasets):
    print("")
    print("There was an error in the program arguments")
    print("There must be 3 arguments: an algorithm, a dataset and a count for how many times it should run")
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
    amount = int(sys.argv[3])

for algorithm in algorithms_to_use:
    for dataset in datasets_to_use:
        best_fscore = 0
        setup_result_folder(algorithm.get_name(), dataset.get_name())
        emails, labels = dataset.load(True)
        glove = GloVe(200)
        # features = glove.get_weights_matrix(emails)
        print("Running algorithm:", algorithm.get_name())
        output_dim = len(set(labels))

        matrix, features_from_matrix = glove.get_weights_matrix(emails, dataset)
        features_from_glove = glove.get_features(emails, datasets_to_use)
        for counter in range(1, amount):


            features = features_from_glove if algorithm.get_name() == "MLP_Tensorflow" else features_from_matrix

            parameters = get_random_params(algorithm.get_name(), features.shape[1], output_dim)

            print("\n#### STARTING RUN NUMBER {} #####\n".format(counter))
            print(str(parameters))
            start_time = time.time()
            if algorithm.get_name() == "MLP_Tensorflow" :
                data_to_plot, y_test, rounded_predictions = algorithm.run_train(dataset, features, labels,
                                                                                parameters)
            else:
                data_to_plot, y_test, rounded_predictions = algorithm.run_train(dataset, features, labels,
                                                                                parameters, embedding=matrix)

            time_taken = time.time() - start_time
            precision, recall, fscore, support = precision_recall_fscore_support(y_test, rounded_predictions)

            avg_fscore = (sum(fscore) / len(fscore))
            print("Avg fScore:", avg_fscore)
            file_path = ROOTPATH + "Results/" + algorithm.get_name() + "/" + dataset.get_name() + "/"
            log_to_file(parameters, fscore, file_path + "resultsfile.csv", time_taken)

            if avg_fscore >= best_fscore:
                best_fscore = avg_fscore
                if not os.path.exists(
                        ROOTPATH + "Results/" + algorithm.get_name() + "/" + dataset.get_name() + "/plots"):
                    os.mkdir(ROOTPATH + "Results/" + algorithm.get_name() + "/" + dataset.get_name() + "/plots")
                #if len(data_to_plot) != 0:
                #    plot_data(data_to_plot[0], file_path + "/plots/" + str(counter) + "_plot_val_acc_.png")
                #    plot_data(data_to_plot[1], file_path + "/plots/" + str(counter) + "_plot_val_loss_.png")
                plot_confusion_matrix(y_test, rounded_predictions, dataset, algorithm, normalize=True,
                                      save_path=file_path + "/plots/" + str(counter) + "_confusmatrix_.png")
