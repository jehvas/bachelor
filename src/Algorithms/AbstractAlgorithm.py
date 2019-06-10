import abc

import time
import uuid

import numpy as np

from rootfile import ROOTPATH
from utility.confusmatrix import plot_confusion_matrix
from utility.plotter import PlotClass, plot_data
from utility.utility import check_directory, log_to_file
import tensorflow as tf

class AbstractAlgorithm(abc.ABC):
    dataset = None
    y_test = None
    predictions = None
    fscore = []
    guid = None

    @abc.abstractmethod
    def prepare_features(self, x_train, y_train, x_test, y_test):
        pass

    @abc.abstractmethod
    def train(self, x_train, y_train, x_test, y_test):
        pass

    @abc.abstractmethod
    def load_parameters(self, parameters):
        pass

    def get_name(self):
        return type(self).__name__

    def plot_data(self, dataset, y_test):
        dataset_name = dataset.get_name()
        file_path = "{}Results/{}/{}/{}/plots/".format(ROOTPATH, dataset.mode, self.get_name(), dataset_name)
        check_directory(file_path)
        if self.get_name() is 'RNN_Tensorflow' or self.get_name() is 'MLP_Tensorflow' or self.get_name() is 'Bi_LSTM_Tensorflow':
            self.plot_graphs(dataset_name, file_path)
        self.plot_matrix(dataset, file_path, y_test)

    def plot_matrix(self, dataset, file_path, y_test):
        plot_confusion_matrix(y_test, self.predictions, dataset, self.get_name(), normalize=True,
                              save_path=file_path + "confusmatrix_" + self.guid + ".png")

    def plot_graphs(self, dataset_name, file_path):
        loss_plot = PlotClass((self.history.epoch, self.history.history["val_loss"]), "Epoch", "Loss",
                              dataset_name, self.get_name())
        plot_data(loss_plot, file_path + "plot_loss_" + self.guid + ".png")

        accuracy_plot = PlotClass((self.history.epoch, self.history.history["val_accuracy"]), "Epoch",
                                  "Accuracy", dataset_name,
                                  self.get_name(), ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        plot_data(accuracy_plot, file_path + "plot_acc_" + self.guid + ".png")

    def write_to_file(self, parameters, time_taken, guid):
        file_path = "{}Results/{}/{}/{}/".format(ROOTPATH, self.dataset.mode, self.get_name(), self.dataset.get_name())
        check_directory(file_path)
        log_to_file(parameters, self.fscore, file_path + "resultsfile.csv", time_taken, guid)

    def run_train(self, dataset, x_train, y_train, x_test, y_test, parameters, should_plot=True):
        start_time = time.time()
        # Generate GUID for each run. If parameter search is run multiple time there is a chance it will risk overriding
        # Plots. Therefor a GUID will also be associated with each run to prevent this.
        self.guid = str(uuid.uuid4())
        self.load_parameters(parameters)
        self.dataset = dataset
        prep = self.prepare_features(x_train, y_train, x_test, y_test)
        if prep is not None:
            x_train, y_train, x_test, y_test = prep
        self.train(x_train, y_train, x_test, y_test)
        print('F-Score:', np.average(self.fscore))
        time_taken = time.time() - start_time
        self.write_to_file(parameters, time_taken, self.guid)
        if should_plot:
            self.plot_data(dataset, y_test)


        # print("Finished in {:.3f}".format(time_taken))
