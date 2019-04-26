import abc
import math
from typing import List

import numpy as np
import tensorflow as tf
import time
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.metrics import Mean, Accuracy
from tensorflow.python.ops.losses.losses_impl import sparse_softmax_cross_entropy
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer

from rootfile import ROOTPATH
from utility.confusmatrix import plot_confusion_matrix
from utility.plotter import plot_data, PlotClass


class AbstractTensorflowAlgorithm(abc.ABC):
    best_fscore_list = []
    fscore_results = []
    train_loss_results = []
    train_accuracy_results = []
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
    loss_function = None
    fscore = []
    dataset = None
    y_test = None
    predictions = None

    def loss(self, x, y):
        y_ = self.model(x)
        return sparse_softmax_cross_entropy(labels=y, logits=y_)

    def grad(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(inputs, targets)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def get_name(self):
        return type(self).__name__

    @abc.abstractmethod
    def generate_model(self):
        pass

    def check_loss(self, loss):
        min_loss = 1e-4
        if math.isnan(loss):
            print('Stopping: Loss is nan!')
            return False
        if loss <= min_loss:
            print('Stopping: Loss is too low!')
            return False
        return True

    def check_fscore(self, epoch, fscore):
        if epoch < 10:
            return True
        else:
            if epoch >= len(self.best_fscore_list):
                return True
            else:
                fscore_is_within_margin = self.best_fscore_list[epoch] - fscore < math.pow(epoch, -0.7)
                if not fscore_is_within_margin:
                    print("Stopping: FScore too low!")
                return fscore_is_within_margin

    def load_parameters(self, parameters):
        self.output_dim = parameters['output_dim']
        self.hidden_dim = parameters['hidden_dim']
        self.input_dim = parameters['input_dim']
        self.input_function = parameters['input_function']
        self.hidden_layers = parameters['hidden_layers']
        self.output_function = parameters['output_function']
        self.optimizer = parameters['optimizer']
        self.loss_function = parameters['loss_function']

    def plot_data(self, dataset_name, counter):
        file_path = ROOTPATH + "Results/" + self.get_name() + "/" + dataset_name + "/"
        self.plot_graphs(dataset_name, counter, file_path)
        self.plot_matrix(counter, file_path)

    def plot_graphs(self, dataset_name, counter, file_path):
        epoch_list = [i for i in range(1, self.epochs_run + 1)]

        loss_plot = PlotClass((epoch_list, self.train_loss_results), "Epoch", "Loss", dataset_name, self.get_name())
        plot_data(loss_plot, file_path + "/plots/" + str(counter) + "_plot_loss_.png")

        accuracy_plot = PlotClass((epoch_list, self.train_accuracy_results), "Epoch", "Accuracy", dataset_name,
                                  self.get_name(), ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        plot_data(accuracy_plot, file_path + "/plots/" + str(counter) + "_plot_acc_.png")

        fscore_plot = PlotClass((epoch_list, self.fscore_results), "Epoch", "Fscore", dataset_name,
                                self.get_name(), ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        plot_data(fscore_plot, file_path + "/plots/" + str(counter) + "_plot_fscore_.png")

    def plot_matrix(self, counter, file_path):
        plot_confusion_matrix(self.y_test, self.predictions, self.dataset, self.get_name(), normalize=True,
                              save_path=file_path + "/plots/" + str(counter) + "_confusmatrix_.png")

    def run_train(self, dataset, features, labels, parameters, embedding=None, best_fscores=None) -> (List, List, List):
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1,
                                                            stratify=labels)
        self.embedding = embedding
        self.best_fscore_list = best_fscores
        self.fscore_results = []
        self.train_loss_results = []
        self.train_accuracy_results = []
        self.load_parameters(parameters)
        self.generate_model()
        self.dataset = dataset
        self.y_test = y_test

        # self.model.summary()

        batch_size = 512

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)  # .shuffle(1024)

        optimizer = self.optimizer

        global_step = tf.Variable(0)

        num_epochs = 30
        print_every = 60
        last_print_time = time.time()
        total_train_entries = len(x_train)
        for epoch in range(num_epochs):
            self.epochs_run = epoch
            epoch_loss_avg = Mean()
            epoch_accuracy = Accuracy()
            # Training loop - using batches of 32
            for x, y in train_dataset:
                if time.time() - last_print_time > print_every:
                    last_print_time = time.time()
                    print("Status: Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, FScore: {:.3f}".format(epoch,
                                                                                                        epoch_loss,
                                                                                                        epoch_accuracy.result(),
                                                                                                        epoch_fscore))
                # Optimize the model
                loss_value, grads = self.grad(x, y)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables),
                                          global_step)

                # Track progress
                epoch_loss_avg(loss_value)  # add current batch loss
                # compare predicted label to actual label
                epoch_accuracy(tf.argmax(self.model(x), axis=1, output_type=tf.int32), y)

            self.predictions = [tf.argmax(x) for x in self.model.predict(x_test)]
            precision, recall, fscore, support = precision_recall_fscore_support(y_test, self.predictions)

            # end epoch
            epoch_loss = epoch_loss_avg.result()
            epoch_fscore = np.average(fscore)
            parameters['Epochs Run'] = epoch + 1
            self.epochs_run = epoch + 1
            self.fscore = fscore
            if not self.check_loss(epoch_loss) or not self.check_fscore(epoch, epoch_fscore):
                print("Loss: {}\tFScore: {}".format(epoch_loss, fscore))
                break
            self.train_loss_results.append(epoch_loss)
            self.fscore_results.append(epoch_fscore)
            self.train_accuracy_results.append(epoch_accuracy.result())
            if epoch % 50 == 0:
                print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, FScore: {:.3f}".format(epoch,
                                                                                            epoch_loss,
                                                                                            epoch_accuracy.result(),
                                                                                            epoch_fscore))


'''
    def train_input_fn(self, features, labels, batch_size):
        """An input function for training"""
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensors([features, labels])

        # Shuffle, repeat, and batch the examples.
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)

        # Return the dataset.
        return dataset
'''
