from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split as tts

from utility.confusmatrix import plot_confusion_matrix

recent_y_test = None
recent_prediction = None
recent_dataset = None

def get_name():
    return 'Perceptron'


def run_train(dataset, features, labels, parameters, embedding=None):
    # Create training data
    x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2)

    model = Perceptron(max_iter=1_000, tol=1e-6)
    print("\nStarting fitting")
    model.fit(x_train, y_train)

    print("Fitting done")
    predictions = model.predict(x_test)

    recent_y_test = y_test
    recent_prediction = predictions
    recent_dataset = dataset

    return [], y_test, predictions


def plot_data(self, dataset_name, counter):
    self.plot_graphs(dataset_name, counter)


def plot_matrix(self):
    plot_confusion_matrix(recent_y_test, recent_prediction, recent_dataset, get_name(), normalize=True)
