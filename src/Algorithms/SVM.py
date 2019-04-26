from sklearn.model_selection import train_test_split as tts
from sklearn.svm import LinearSVC

from utility.confusmatrix import plot_confusion_matrix


recent_y_test = None
recent_prediction = None
recent_dataset = None

def get_name():
    return 'SVM'


def run_train(dataset, features, labels, parameters, embedding=None):
    print("Running algorithm: Algorithms.SVM")
    # Create training data
    x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2, random_state=1)

    # Algorithms.SVM Stuff
    svm_classifier = LinearSVC(loss=parameters['loss_function'], class_weight=parameters['class_weights'], penalty=parameters['penalty'])

    print("\nStarting fitting")
    svm_classifier.fit(x_train, y_train)

    print("Fitting done")
    predictions = svm_classifier.predict(x_test)

    recent_y_test = y_test
    recent_prediction = predictions
    recent_dataset = dataset

    return [], y_test, predictions

    parameters


def plot_data(self, dataset_name, counter):
    self.plot_graphs(dataset_name, counter)


def plot_matrix(self):
    plot_confusion_matrix(recent_y_test, recent_prediction, recent_dataset, get_name(), normalize=True)
