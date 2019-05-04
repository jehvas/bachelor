import abc
import uuid
from typing import List

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.python import set_random_seed
from tensorflow.python.keras.callbacks import EarlyStopping

from rootfile import ROOTPATH
from utility.confusmatrix import plot_confusion_matrix
from utility.plotter import plot_data, PlotClass


class AbstractTensorflowAlgorithm(abc.ABC):
    epochs_run = 0
    embedding = []
    model = None
    output_dim = None
    hidden_dim = None
    input_dim = None
    num_epochs = None
    batch_size = None
    input_function = None
    hidden_layers = None
    output_function = None
    optimizer = None
    fscore = []
    dataset = None
    y_test = None
    predictions = None
    guid = None
    history = None

    def get_name(self):
        return type(self).__name__

    @abc.abstractmethod
    def generate_model(self, middle_layers, input_shape):
        pass

    def load_parameters(self, parameters):
        self.output_dim = parameters['output_dim']
        self.hidden_dim = parameters['hidden_dim']
        self.input_dim = parameters['input_dim']
        self.input_function = parameters['input_function']
        self.hidden_layers = parameters['hidden_layers']
        self.output_function = parameters['output_function']
        self.optimizer = parameters['optimizer']

    def plot_data(self, dataset_name, counter):
        file_path = ROOTPATH + "Results/" + self.get_name() + "/" + dataset_name + "/"
        self.plot_graphs(dataset_name, counter, file_path)
        self.plot_matrix(counter, file_path)

    def plot_graphs(self, dataset_name, counter, file_path):

        loss_plot = PlotClass((self.history.epoch, self.history.history["val_loss"]), "Epoch", "Loss",
                              dataset_name, self.get_name())
        plot_data(loss_plot, file_path + "/plots/" + str(counter) + "_plot_loss_" + self.guid + ".png")

        accuracy_plot = PlotClass((self.history.epoch, self.history.history["val_accuracy"]), "Epoch",
                                  "Accuracy", dataset_name,
                                  self.get_name(), ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        plot_data(accuracy_plot, file_path + "/plots/" + str(counter) + "_plot_acc_" + self.guid + ".png")

    def plot_matrix(self, counter, file_path):
        plot_confusion_matrix(self.y_test, self.predictions, self.dataset, self.get_name(), normalize=True,
                              save_path=file_path + "/plots/" + str(counter) + "_confusmatrix_" + self.guid + ".png")

    def run_train(self, dataset, train_data, test_data, parameters, embedding=None) -> (
            List, List, List):

        x_train, y_train = train_data
        x_test, y_test = test_data
        set_random_seed(1)


        self.embedding = embedding
        self.load_parameters(parameters)
        self.dataset = dataset
        self.y_test = y_test
        if self.get_name() != "MLP_Tensorflow":
            x_train = np.expand_dims(x_train, axis=1)
            y_train = np.expand_dims(y_train, axis=1)
            x_test = np.expand_dims(x_test, axis=1)
            y_test = np.expand_dims(y_test, axis=1)
        else:
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            x_test = np.array(x_test)
            y_test = np.array(y_test)

        # Generate GUID for each run. If parameter search is run multiple time there is a chance it will risk overriding
        # Plots. Therefor a GUID will also be associated with each run to prevent this.
        self.guid = str(uuid.uuid4())

        self.generate_model(self.hidden_layers, x_train.shape[1:])
        opt = parameters["optimizer"]

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'],
        )
        es_loss = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)

        self.history = self.model.fit(x_train,
                                      y_train,
                                      epochs=500,
                                      callbacks=[es_loss],
                                      validation_data=(x_test, y_test),
                                      verbose=0)

        self.predictions = self.model.predict(x_test)

        self.predictions = tf.argmax(self.predictions, axis=1)
        precision, recall, fscore, support = precision_recall_fscore_support(y_test, self.predictions)
        print(np.average(fscore))

        self.epochs_run = len(self.history.epoch)
        parameters['Epochs Run'] = self.epochs_run
        self.fscore = fscore


def print_status(epoch, loss, accuracy, fscore):
    print("Status: Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, FScore: {:.3f}".format(epoch,
                                                                                        loss,
                                                                                        accuracy,
                                                                                        fscore))
