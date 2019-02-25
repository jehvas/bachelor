import pickle as c
import os
from collections import Counter

from pip._vendor.distlib.compat import raw_input


def load(svm_file):
    with open(svm_file, "rb") as fp:
        svm = c.load(fp)
    return svm


svm = load("text-classifier.mdl")


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


d = make_dict()



while True:
    features = []
    inp = raw_input(">").split()
    if inp[0] == "exit":
        break
    for word in d:
        features.append(inp.count(word[0]))
    res = svm.predict([features])
    print(["Ham", "Spam!"][res[0]])