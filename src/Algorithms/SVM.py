import uuid

from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import LinearSVC

from Algorithms.AbstractAlgorithm import AbstractAlgorithm
from rootfile import ROOTPATH
from utility.confusmatrix import plot_confusion_matrix


class SVM(AbstractAlgorithm):
    def prepare_features(self, x_train, y_train, x_test, y_test):
        return None

    def load_parameters(self, parameters):
        self.loss_function = parameters['loss_function']

    fscore = None
    loss_function = None

    def train(self, x_train, y_train, x_test, y_test):
        clf = LinearSVC(loss=self.loss_function, class_weight='balanced')
        clf.fit(x_train, y_train)
        predictions = clf.predict(x_test)

        self.predictions = predictions

        precision, recall, fscore, support = precision_recall_fscore_support(y_test, predictions)
        print(sum(fscore) / len(fscore))
        self.fscore = fscore
        return [], y_test, predictions
