import abc

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.python.keras.callbacks import EarlyStopping

from Algorithms.AbstractAlgorithm import AbstractAlgorithm


class AbstractTensorflowAlgorithm(AbstractAlgorithm):
    epochs_run = 0
    model = None
    hidden_layers = None
    optimizer = None
    predictions = None
    history = None

    def get_name(self):
        return type(self).__name__

    @abc.abstractmethod
    def generate_model(self, middle_layers, input_shape, output_dim):
        pass

    def load_parameters(self, parameters):
        self.hidden_layers = parameters['hidden_layers']
        self.optimizer = parameters['optimizer']

    def train(self, x_train, y_train, x_test, y_test):
        self.generate_model(self.hidden_layers, x_train.shape[1:], len(self.dataset.classes))

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=self.optimizer,
            metrics=['accuracy'],
        )
        es_loss = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True)
        self.history = self.model.fit(x_train,
                                      y_train,
                                      epochs=50,
                                      callbacks=[es_loss],
                                      validation_data=(x_test, y_test),
                                      verbose=0)

        self.predictions = self.model.predict(x_test)

        self.predictions = tf.argmax(self.predictions, axis=1)
        precision, recall, fscore, support = precision_recall_fscore_support(y_test, self.predictions)
        print(np.average(fscore))

        self.epochs_run = len(self.history.epoch)
        # parameters['Epochs Run'] = self.epochs_run
        self.fscore = fscore


def print_status(epoch, loss, accuracy, fscore):
    print("Status: Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, FScore: {:.3f}".format(epoch,
                                                                                        loss,
                                                                                        accuracy,
                                                                                        fscore))
