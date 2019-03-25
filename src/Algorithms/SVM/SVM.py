from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split as tts
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split as tts, StratifiedKFold
from sklearn.svm import SVC

from DatasetsConsumers.Chromium import Chromium
from DatasetsConsumers.Newsgroups import Newsgroups
from Glove.glovemodel import GloVe

print("Running algorithm: Algorithms.SVM")

# Load dataset
Dataset_Consumer = Newsgroups()
emails, labels = Dataset_Consumer.load(True)

# Load GloVe model
GloVe_Obj = GloVe(50)
features = GloVe_Obj.get_features(emails, Dataset_Consumer)

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
