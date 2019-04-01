from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split as tts
from sklearn.svm import SVC, LinearSVC

from DatasetsConsumers.Chromium import Chromium
from DatasetsConsumers.Newsgroups import Newsgroups
from DatasetsConsumers.SpamHam import SpamHam
from Glove.glovemodel import GloVe

print("Running algorithm: Algorithms.SVM")

# Load dataset
Dataset_Consumer = SpamHam()
emails, labels = Dataset_Consumer.load(True)

# Load GloVe model
GloVe_Obj = GloVe(200)
features = GloVe_Obj.get_features(emails, Dataset_Consumer)

# Create training data
x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2)


# Algorithms.SVM Stuff
Perceptron = Perceptron()


print("\nStarting fitting")
Perceptron.fit(x_train, y_train)

print("Fitting done")
predictions = Perceptron.predict(x_test)

precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, predictions)
print("\n--- Results ---")
print("Precision: ", precision)
print()
print("\n\nRecall: ", recall)
print("\n\nF_score: ", fbeta_score)
