from Algorithms.AbstractTensorflowAlgorithm import AbstractTensorflowAlgorithm
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Embedding, Dense
from utility.model_factory import make_hidden_layers


class Bi_LSTM_Tensorflow(AbstractTensorflowAlgorithm):
    def generate_model(self):
        self.model = Sequential(
            [Embedding(self.embedding.shape[0], self.embedding.shape[1], weights=[self.embedding],
                       input_length=self.input_dim)] +
            make_hidden_layers(self.hidden_dim, self.hidden_layers) +
            [Dense(self.output_dim, name='out_layer')]
        )


'''
def learning_rate_function(epoch, learning_rate):
    return learning_rate * 0.99


def get_name():
    return 'MLP_Tensorflow'


def run_train(dataset, features, labels, parameters, embedding=None) -> (List, List, List):
    x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2, random_state=1, stratify=labels)

    output_dim = parameters['output_dim']
    hidden_dim = parameters['hidden_dim']
    input_dim = parameters['input_dim']
    num_epochs = parameters['num_epochs']
    batch_size = parameters['batch_size']
    input_function = parameters['input_function']
    hidden_layers = parameters['hidden_layers']
    output_function = parameters['output_function']
    optimizer = parameters['optimizer']
    loss_function = parameters['loss_function']

    def MLP():
        model = generate_mlp_model(input_dim, hidden_dim, hidden_layers, output_dim, input_function, output_function)
        return model

    mlp_model = MLP()
    mlp_model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
    mlp_model.summary()
    history = mlp_model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
                            validation_data=(x_test, y_test), workers=4, verbose=1,
                            callbacks=[LearningRateScheduler(learning_rate_function, verbose=1),
                                       EarlyStopping(monitor='loss', min_delta=0, patience=1, verbose=1,
                                                     mode='auto',
                                                     restore_best_weights=True),
                                       TerminateOnNaN()
                                       ])
    epochs_finished = len(history.history['val_accuracy'])
    parameters['num_epochs'] = epochs_finished

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
'''


'''
def learning_rate_function(epoch, learning_rate):
    return learning_rate * 0.99


def get_name() -> str:
    return 'Bi-LSTM_Tensorflow'


def run_train(dataset, features, labels, parameters, embedding=None) -> (List, List, List):
    x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2, random_state=1, stratify=labels)

    output_dim = parameters['output_dim']
    hidden_dim = parameters['hidden_dim']
    input_dim = parameters['input_dim']
    num_epochs = parameters['num_epochs']
    batch_size = parameters['batch_size']
    input_function = parameters['input_function']
    hidden_layers = parameters['hidden_layers']
    output_function = parameters['output_function']
    optimizer = parameters['optimizer']
    loss_function = parameters['loss_function']

    def Bi_LSTM():
        model = generate_bi_lstm_model(input_dim, hidden_dim, hidden_layers, output_dim, input_function,
                                       output_function, embedding)
        return model

    bi_lstm_model = Bi_LSTM()
    bi_lstm_model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    history = bi_lstm_model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
                                validation_data=(x_test, y_test), workers=4,
                                callbacks=[LearningRateScheduler(learning_rate_function, verbose=1),
                                           EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=1, verbose=1,
                                                         mode='auto',
                                                         restore_best_weights=True),
                                           TerminateOnNaN()
                                           ])
    epochs_finished = len(history.history['val_accuracy'])
    parameters['num_epochs'] = epochs_finished

    iteration_list = [i for i in range(1, num_epochs + 1)]

    predictions = bi_lstm_model.predict(x_test)
    rounded_predictions = [np.argmax(x) for x in predictions]

    accr = bi_lstm_model.evaluate(x_test, y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
    return ([
        PlotClass([(iteration_list, history.history['val_accuracy'])], "Epoch", "Accuracy", parameters, dataset,
                  "Bi-LSTM",
                  legend=(['train', 'test'], 'upper left')),
        PlotClass([(iteration_list, history.history['val_loss'])], "Epoch", "Loss", parameters, dataset, "Bi-LSTM",
                  legend=(['train', 'test'], 'upper left'))
    ]), y_test, rounded_predictions
'''