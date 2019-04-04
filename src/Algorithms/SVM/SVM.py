from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split as tts
from sklearn.svm import LinearSVC


def run_train(dataset, features, labels, parameters):
    print("Running algorithm: Algorithms.SVM")
    # Create training data
    x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2, random_state=1)

    # Algorithms.SVM Stuff
    svm_classifier = LinearSVC()

    print("\nStarting fitting")
    svm_classifier.fit(x_train, y_train)

    print("Fitting done")
    predictions = svm_classifier.predict(x_test)

    precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, predictions)
    print("\n--- Results ---")
    print("Precision: ", precision)
    print()
    print("\n\nRecall: ", recall)
    print("\n\nF_score: ", fbeta_score)
