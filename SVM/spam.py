import math
import os

import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split as tts
from sklearn.svm import SVC

from utility import utility

word_indexes = {}


def filter_emails():
    direc = "emails/"
    files = os.listdir(direc)

    emails = [direc + email for email in files]
    words = []
    ec = len(emails)
    labels = []
    for email in emails:
        if (ec % 100 == 0):
            print(ec)
        ec = ec - 1
        f = open(email, encoding="latin-1")
        text = f.read()
        f.close()

        words.append(process_single_mail(text))

        if "ham" in email:
            labels.append(0)
        if "spam" in email:
            labels.append(1)
    return words, labels


def process_single_mail(text):
    texttokenized = word_tokenize(text.lower())
    sentence_no_stopwords = filterStopWords(texttokenized)
    email_words = [w for w in sentence_no_stopwords if w.isalpha()]
    return email_words


def sum_vectors(words_in_emails):
    all_vector_sum = []
    for words in words_in_emails:
        vector_sum = np.zeros(dimensionCount)
        for i in range(len(words)):
            for j in range(dimensionCount):
                if words[i] in gModel:
                    vector_sum[j] += gModel[words[i]][j]
        all_vector_sum.append(vector_sum)
    return all_vector_sum


def root_mean_square(vectors_sum_2darray):
    rms_array = []
    for vector_sum in vectors_sum_2darray:
        rms = 0
        for i in range(len(vector_sum)):
            rms = rms + vector_sum[i] * vector_sum[i]
        rms = math.sqrt(rms / dimensionCount)
        rms_array.append(rms)
    return rms_array


def create_sentence_vector(rms_array, vectors_sum_2darray):
    sentence_vectors = vectors_sum_2darray
    for i in range(len(vectors_sum_2darray)):
        for j in range(len(vectors_sum_2darray[i])):
            sentence_vectors[i][j] = vectors_sum_2darray[i][j] / rms_array[i]
    return sentence_vectors


stop_words = set(stopwords.words("english"))


def filterStopWords(texttokenized):
    filtered_sentence = []
    for w in texttokenized:
        if w not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence


gModel = utility.load("saved_gModel")
dimensionCount = len(next(iter(gModel.values())))
emails, labels = filter_emails()
utility.save(emails, "saved_emails")
utility.save(labels, "saved_labels")
#emails = utility.load("saved_emails")
#labels = utility.load("saved_labels")
sum_vectors_array = sum_vectors(emails)
rms_array = root_mean_square(sum_vectors_array)
features = create_sentence_vector(rms_array, sum_vectors_array)

x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2)

svclassifier = SVC(kernel="linear")
print("\nStarting fitting")
svclassifier.fit(x_train, y_train)
print("Fitting done")
preds = svclassifier.predict(x_test)

precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, preds)
print("\nResults")
print("Precision: ", precision)
print("Recall: ", recall)
print("F_score: ", fbeta_score)
# print(accuracy_score(y_test, preds))
# save(svclassifier, "text-classifier.mdl")
