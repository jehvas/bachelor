from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split as tts


def get_name():
    return 'Perceptron'


def run_train(dataset, features, labels, parameters, matrix, sequences_matrix, emails):
    # Create training data
    x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2)

    model = Perceptron()

    print("\nStarting fitting")
    model.fit(x_train, y_train)

    print("Fitting done")
    predictions = model.predict(x_test)

    return [], y_test, predictions
