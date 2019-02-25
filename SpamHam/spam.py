import os
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
import pickle as c


def save(clf, name):
    with open(name, 'wb') as fp:
        c.dump(clf, fp)
    print("saved")


def make_dict():
    direc = "emails/"
    files = os.listdir(direc)

    emails = [direc + email for email in files]
    words = []
    for email in emails:
        f = open(email, encoding="latin-1")
        text = f.read()
        words += text.split(" ")

    for i in range(len(words)):
        if not words[i].isalpha():
            words[i] = ""

    dict = Counter(words)
    del dict[""]
    return dict.most_common(3000)


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
        data = []
        f = open(email, encoding="latin-1")
        words = f.read().split(" ")
        for entry in dict:
            data.append(words.count(entry[0]))
        feature_set.append(data);
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

#svclassifier = SVC(kernel="linear")
#svclassifier.fit(x_train, y_train)
#preds = svclassifier.predict(x_test)
clf = MultinomialNB()
clf.fit(x_train, y_train)
preds = clf.predict(x_test)

print(accuracy_score(y_test, preds))
save(clf, "text-classifier.mdl")