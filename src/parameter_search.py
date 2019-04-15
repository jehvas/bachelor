import os
import random
import time

from sklearn.metrics import precision_recall_fscore_support

from Algorithms.MLPT import MLP_tensorflow
from Algorithms.Perceptron import Perceptron
from Algorithms.SVM import SVM
from DatasetsConsumers.Newsgroups import Newsgroups
from DatasetsConsumers.Spamassassin import Spamassassin
from Glove.glovemodel import GloVe
from rootfile import ROOTPATH
from utility import confusmatrix
from utility.Random_Parameters import get_random_params
from utility.confusmatrix import plot_confusion_matrix
from utility.plotter import plot_data
from utility.utility import log_to_file, setup_result_folder





best_fscore = 0.0
counter = 1
dataset_consumer = Newsgroups()
algorithm = Perceptron

setup_result_folder(algorithm.get_name(), dataset_consumer.get_name())

emails, labels = dataset_consumer.load(True)
glove = GloVe(200)
features = glove.get_features(emails, dataset_consumer)
print("Running algorithm:", algorithm.get_name())
while True:
    output_dim = len(set(labels))
    parameters = get_random_params(algorithm.get_name(), features.shape[1], output_dim)

    print("\n#### STARTING RUN NUMBER {} #####\n".format(counter))
    print(str(parameters))
    start_time = time.time()
    data_to_plot, y_test, rounded_predictions = algorithm.run_train(dataset_consumer, features, labels,
                                                                    parameters)
    time_taken = time.time() - start_time
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, rounded_predictions)
    # print("\nPrecision: ", precision)
    # print("\nRecall: ", recall)
    # print("\nFscore: ", fscore)
    # print("\n")
    avg_fscore = (sum(fscore) / len(fscore))
    print("Avg fScore:", avg_fscore)
    file_path = ROOTPATH + "Results/" + algorithm.get_name() + "/" + dataset_consumer.get_name() + "/"
    log_to_file(parameters, fscore, file_path + "resultsfile.csv", time_taken)

    if avg_fscore >= best_fscore:
        best_fscore = avg_fscore
        if not os.path.exists(ROOTPATH + "Results/" + algorithm.get_name() + "/" + dataset_consumer.get_name() + "/plots"):
            os.mkdir(ROOTPATH + "Results/" + algorithm.get_name() + "/" + dataset_consumer.get_name() + "/plots")
        if len(data_to_plot) != 0:
            plot_data(data_to_plot[0], file_path + "/plots/" + str(counter) + "_plot_val_acc_.png")
            plot_data(data_to_plot[1], file_path + "/plots/" + str(counter) + "_plot_val_loss_.png")
        plot_confusion_matrix(y_test, rounded_predictions, dataset_consumer, algorithm, normalize=True,
                              save_path=file_path + "/plots/" + str(counter) + "_confusmatrix_.png")

    counter += 1
