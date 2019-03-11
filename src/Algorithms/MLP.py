from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split as tts
from DatasetsConsumers.Chromium import Chromium
from DatasetsConsumers.Enron import Enron
from Glove.glovemodel import GloVe

print("Running algorithm: MLP")

# Load dataset
Dataset_Consumer = Enron()
emails, labels = Dataset_Consumer.load(True)

# Load GloVe model
GloVe_Obj = GloVe("glove.6B.50d.txt")
features = GloVe_Obj.get_features(emails)

# Create training data & SVM Stuff
x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2)


# MLP Stuff
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

print("\nStarting fitting")
clf.fit(x_train, y_train)

print("Fitting done")
predictions = clf.predict(x_test)

precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, predictions)
print("\n--- Results ---")
print("Precision: ", precision)
print()
print("\n\nRecall: ", recall)
print("\n\nF_score: ", fbeta_score)