import abc
import math
from typing import List

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.metrics import Mean, Accuracy
from tensorflow.python.ops.losses.losses_impl import sparse_softmax_cross_entropy
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer


class AbstractTensorflowAlgorithm(abc.ABC):
    best_fscore_list = []
    fscore_results = []
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

    def run_train(self, dataset, features, labels, parameters, embedding=None, best_fscores=None) -> (List, List, List):
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1,
                                                            stratify=labels)
        self.embedding = embedding
        self.best_fscore_list = best_fscores
        self.load_parameters(parameters)
        self.generate_model()
        # self.model.summary()

        batch_size = 320

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)  # .shuffle(1024)

        optimizer = self.optimizer
        # optimizer = GradientDescentOptimizer(learning_rate=0.01)

        global_step = tf.Variable(0)

        # keep results for plotting
        train_loss_results = []
        train_accuracy_results = []
        total_epochs = 0

        num_epochs = 30
        total_train_entries = len(x_train)
        for epoch in range(num_epochs):
            self.epochs_run = epoch
            epoch_loss_avg = Mean()
            epoch_accuracy = Accuracy()
            # progress = 0
            # Training loop - using batches of 32
            for x, y in train_dataset:
                # progress += len(x)
                # print("{} / {}".format(progress, total_train_entries))
                # Optimize the model
                loss_value, grads = self.grad(x, y)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables),
                                          global_step)

                # Track progress
                epoch_loss_avg(loss_value)  # add current batch loss
                # compare predicted label to actual label
                epoch_accuracy(tf.argmax(self.model(x), axis=1, output_type=tf.int32), y)

            predictions = [tf.argmax(x) for x in self.model.predict(x_test)]
            precision, recall, fscore, support = precision_recall_fscore_support(y_test, predictions)

            # end epoch
            epoch_loss = epoch_loss_avg.result()
            epoch_fscore = np.average(fscore)
            parameters['Epochs Run'] = epoch + 1
            self.fscore = fscore
            if not self.check_loss(epoch_loss) or not self.check_fscore(epoch, epoch_fscore):
                print("Early stopping Loss: {}\tFScore: {}".format(epoch_loss, fscore))
                break
            train_loss_results.append(epoch_loss)
            self.fscore_results.append(epoch_fscore)
            train_accuracy_results.append(epoch_accuracy.result())
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
