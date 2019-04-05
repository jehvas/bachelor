from sklearn.model_selection import train_test_split as tts
from sklearn.svm import LinearSVC


def get_name():
    return 'SVM'


def run_train(dataset, features, labels, parameters):
    print("Running algorithm: Algorithms.SVM")
    # Create training data
    x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2, random_state=1)

    # Algorithms.SVM Stuff
    svm_classifier = LinearSVC(class_weight=parameters['class_weights'])

    print("\nStarting fitting")
    svm_classifier.fit(x_train, y_train)

    print("Fitting done")
    predictions = svm_classifier.predict(x_test)

    return [], y_test, predictions
