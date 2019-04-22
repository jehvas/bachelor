from typing import List
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split as tts
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Flatten, Dense, Activation, Dropout, Embedding, ELU, PReLU, ReLU, Softmax
from tensorflow.python.keras.optimizers import Adam

from utility.model_factory import generate_model
from utility.plotter import PlotClass


def get_name():
    return 'MLP_Tensorflow'


def run_train(dataset, features, labels, parameters, embedding=None) -> (List, List, List):
    x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2, random_state=1, stratify=labels)

    output_dim = parameters['output_dim']
    hidden_dim = parameters['hidden_dim']
    input_dim = parameters['input_dim']
    # max_len = parameters['max_len']
    num_epochs = parameters['num_epochs']
    batch_size = parameters['batch_size']
    input_function = parameters['input_function']
    hidden_layers = parameters['hidden_layers']
    output_function = parameters['output_function']

    def MLP():
        model = generate_model(input_dim, hidden_dim, hidden_layers, output_dim, input_function, output_function)
        return model

    mlp_model = MLP()
    mlp_model.compile(loss='sparse_categorical_crossentropy', optimizer=parameters['optimizer'], metrics=['accuracy'])

    history = mlp_model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
                            validation_data=(x_test, y_test), workers=4, verbose=0)

    iteration_list = [i for i in range(1, num_epochs + 1)]

    predictions = mlp_model.predict(x_test)
    rounded_predictions = [np.argmax(x) for x in predictions]

    accr = mlp_model.evaluate(x_test, y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
    return ([
        PlotClass([(iteration_list, history.history['val_accuracy'])], "Epoch", "Accuracy", parameters, dataset, "MLP",
                  legend=(['train', 'test'], 'upper left')),
        PlotClass([(iteration_list, history.history['val_loss'])], "Epoch", "Loss", parameters, dataset, "MLP",
                  legend=(['train', 'test'], 'upper left'))
    ]), y_test, rounded_predictions
