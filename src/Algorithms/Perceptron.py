import uuid

from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_recall_fscore_support

from Algorithms.AbstractAlgorithm import AbstractAlgorithm
from rootfile import ROOTPATH
from utility.confusmatrix import plot_confusion_matrix


class Perceptron(AbstractAlgorithm):
    def prepare_features(self, x_train, y_train, x_test, y_test):
        return None

    def load_parameters(self, parameters):
        self.penalty = parameters['penalty']

    fscore = None
    penalty = None

    def train(self, x_train, y_train, x_test, y_test):
        model = Perceptron(max_iter=1_000, tol=1e-6, class_weight='balanced', penalty=self.penalty)
        model.fit(x_train, y_train)

        predictions = model.predict(x_test)
        self.predictions = predictions

        precision, recall, fscore, support = precision_recall_fscore_support(y_test, predictions)
        self.fscore = fscore
        return [], y_test, predictions

    def plot_data(self, dataset, counter, dataset_mode, y_test):
        file_path = ROOTPATH + "Results/" + dataset_mode + "/" + self.get_name() + "/" + dataset.get_name() + "/"
        self.plot_matrix(dataset, counter, file_path, y_test)
