import math
import os

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split as tts
from sklearn.svm import SVC

from DatasetsConsumers import SpamHam
from Glove import glovemodel
from src.utility import utility

word_indexes = {}


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
            if rms_array[i] == 0:
                sentence_vectors[i][j] = 0
            else:
                sentence_vectors[i][j] = vectors_sum_2darray[i][j] / rms_array[i]
    return sentence_vectors


if os.path.exists(utility.output_path + "Glove_saved_gModel"):
    gModel = utility.load(utility.output_path + "Glove_saved_gModel")
else:
    gModel = glovemodel.loadGloveModel("../../data/GloveWordEmbeddings/glove.6B.50d.txt")
    utility.save(gModel, utility.output_path + "Glove_saved_gModel")

dimensionCount = len(next(iter(gModel.values())))

Dataset_Consumer = SpamHam.SpamHam()

emails, labels = Dataset_Consumer.load(True)
# utility.save(emails, "saved_emails")
# utility.save(labels, "saved_labels")
# emails = utility.load("saved_emails")
# labels = utility.load("saved_labels")
sum_vectors_array = sum_vectors(emails)
rms_array = root_mean_square(sum_vectors_array)
features = create_sentence_vector(rms_array, sum_vectors_array)

x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2)

svmclassifier = SVC(kernel="linear")
print("\nStarting fitting")
svmclassifier.fit(x_train, y_train)
print("Fitting done")
preds = svmclassifier.predict(x_test)

precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, preds)
print("\nResults")
print("Precision: ", precision)
print("Recall: ", recall)
print("F_score: ", fbeta_score)
# print(accuracy_score(y_test, preds))
# save(svclassifier, "text-classifier.mdl")
