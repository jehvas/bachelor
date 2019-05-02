import uuid

from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import LinearSVC

from rootfile import ROOTPATH
from utility.confusmatrix import plot_confusion_matrix


recent_y_test = None
recent_predictions = None
recent_dataset = None
fscore = None
guid = None

def get_name():
    return 'SVM'


def run_train(dataset, train_data, test_data, parameters, embedding=None, best_fscores=None):
    x_train, y_train = train_data
    x_test, y_test = test_data
    # Algorithms.SVM Stuff
    svm_classifier = LinearSVC(loss=parameters['loss_function'], class_weight=parameters['class_weights'], penalty=parameters['penalty'])

    print("\nStarting fitting")
    svm_classifier.fit(x_train, y_train)

    print("Fitting done")
    predictions = svm_classifier.predict(x_test)

    global recent_y_test
    recent_y_test = y_test
    global recent_predictions
    recent_predictions = predictions
    global recent_dataset
    recent_dataset = dataset

    precision, recall, _fscore, support = precision_recall_fscore_support(y_test, predictions)
    global fscore
    fscore = _fscore
    global guid
    guid = str(uuid.uuid4())
    return [], y_test, predictions

    parameters


def plot_data(dataset_name, counter):
    file_path = ROOTPATH + "Results/" + get_name() + "/" + recent_dataset.get_name() + "/"
    plot_matrix(counter, file_path)


def plot_matrix(counter, file_path):
    plot_confusion_matrix(recent_y_test, recent_predictions, recent_dataset, get_name(), normalize=True,
                          save_path=file_path + "/plots/" + str(counter) + "_confusmatrix_" + guid + ".png")
