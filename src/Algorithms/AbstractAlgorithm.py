import abc

import time
import uuid

from rootfile import ROOTPATH
from utility.confusmatrix import plot_confusion_matrix
from utility.plotter import PlotClass, plot_data


class AbstractAlgorithm(abc.ABC):
    dataset = None
    guid = None
    y_test = None
    predictions = None

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

    def plot_data(self, dataset, counter, dataset_mode, y_test):
        dataset_name = dataset.get_name()
        file_path = ROOTPATH + "Results/" + dataset_mode + "/" + self.get_name() + "/" + dataset_name + "/"
        if dataset_name is 'RNN_Tensorflow' or dataset_name is 'MLP_Tensorflow' or dataset_name is 'Bi_LSTM_Tensorflow':
            self.plot_graphs(dataset_name, counter, file_path)
        self.plot_matrix(dataset_name, counter, file_path, y_test)

    def plot_matrix(self, dataset, counter, file_path, y_test):
        plot_confusion_matrix(y_test, self.predictions, dataset, self.get_name(), normalize=True,
                              save_path=file_path + "/plots/" + str(counter) + "_confusmatrix_" + self.guid + ".png")

    def plot_graphs(self, dataset_name, counter, file_path):
        loss_plot = PlotClass((self.history.epoch, self.history.history["val_loss"]), "Epoch", "Loss",
                              dataset_name, self.get_name())
        plot_data(loss_plot, file_path + "/plots/" + str(counter) + "_plot_loss_" + self.guid + ".png")

        accuracy_plot = PlotClass((self.history.epoch, self.history.history["val_accuracy"]), "Epoch",
                                  "Accuracy", dataset_name,
                                  self.get_name(), ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        plot_data(accuracy_plot, file_path + "/plots/" + str(counter) + "_plot_acc_" + self.guid + ".png")

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

        time_taken = time.time() - start_time
        print("Finished in {:.3f}".format(time_taken))
        pass
