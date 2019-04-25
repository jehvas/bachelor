from typing import List

import numpy as np
from sklearn.model_selection import train_test_split as tts
from tensorflow.python.keras.optimizers import RMSprop

from utility.model_factory import generate_bi_lstm_model
from utility.plotter import PlotClass


def get_name() -> str:
    return 'Bi-LSTM_Tensorflow'


def run_train(dataset, features, labels, parameters, embedding=None) -> (List, List, List):
    x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2, random_state=1, stratify=labels)

    output_dim = parameters['output_dim']
    hidden_dim = parameters['hidden_dim']
    input_dim = parameters['input_dim']
    hidden_layers = parameters['hidden_layers']
    num_epochs = parameters['num_epochs']
    batch_size = parameters['batch_size']
    input_function = parameters['input_function']
    output_function = parameters['output_function']

    def Bi_LSTM():
        model = generate_bi_lstm_model(input_dim, hidden_dim, hidden_layers, output_dim, input_function, output_function, embedding)
        return model

    bi_lstm_model = Bi_LSTM()
    bi_lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    history = bi_lstm_model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
                                validation_data=(x_test, y_test), workers=4)

    iteration_list = [i for i in range(1, num_epochs + 1)]

    predictions = bi_lstm_model.predict(x_test)
    rounded_predictions = [np.argmax(x) for x in predictions]

    accr = bi_lstm_model.evaluate(x_test, y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
    return ([
        PlotClass([(iteration_list, history.history['val_accuracy'])], "Epoch", "Accuracy", parameters, dataset, "Bi-LSTM",
                  legend=(['train', 'test'], 'upper left')),
        PlotClass([(iteration_list, history.history['val_loss'])], "Epoch", "Loss", parameters, dataset, "Bi-LSTM",
                  legend=(['train', 'test'], 'upper left'))
    ]), y_test, rounded_predictions
