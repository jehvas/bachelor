import os
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
import pickle as c
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def save(clf, name):
    with open(name, 'wb') as fp:
        c.dump(clf, fp)
    print("saved")

word_indexes = {}
def make_dict():
    direc = "emails/"
    files = os.listdir(direc)

    emails = [direc + email for email in files]
    words = []
    for email in emails:
        f = open(email, encoding="latin-1")
        text = f.read().lower()
        words += text.split(" ")

    for i in range(len(words)):
        if not words[i].isalpha():
            words[i] = ""

    dict = Counter(words)
    del dict[""]
    for x in [x for x in dict.keys() if dict[x] in range(20, 500)]:
        dict.pop(x)
    dict = dict.most_common(3000)
    for entry in dict:
        if entry[0] not in word_indexes:
            word_indexes[entry[0]] = len(word_indexes)

    return dict


def make_dataset(dict):
    direc = "emails/"
    files = os.listdir(direc)

    feature_set = []
    labels = []
    emails = [direc + email for email in files]
    c = len(emails)
    for email in emails:
        if (c % 100) == 0:
            print(c)
        data = np.ndarray.tolist(np.zeros(len(dict), dtype=np.int32))
        f = open(email, encoding="latin-1")
        words = f.read().split(" ")
        for word in words:
            word = word.lower()
            if word in word_indexes:
                i = word_indexes[word]
                data[i] = data[i]+1
        #for entry in dict:
        #    data.append(words.count(entry[0]))
        feature_set.append(data)
        if "ham" in email:
            labels.append(0)
        if "spam" in email:
            labels.append(1)
        c = c-1
    return feature_set, labels


d = make_dict()
save(d, "spamdict.dict")
features, labels = make_dataset(d)

x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2)

#pca = PCA(n_components=2)
#pca.fit(x_train[0])

#titles = ('SVC with linear kernel', 'LinearSVC (linear kernel)', 'SVC with RBF kernel', 'SVC with polynomial (degree 3) kernel')
#fig, sub = plt.subplots(2, 2)
#plt.subplots_adjust(wspace=0.4, hspace=0.4)
#print(pca.singular_values_)
#plt.show()

svclassifier = SVC(kernel="linear")
svclassifier.fit(x_train, y_train)
preds = svclassifier.predict(x_test)
#clf = MultinomialNB()
#clf.fit(x_train, y_train)
#preds = clf.predict(x_test)


print(accuracy_score(y_test, preds))
save(svclassifier, "text-classifier.mdl")