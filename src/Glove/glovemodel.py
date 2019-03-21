import numpy as np
import os

import time
import torch

from utility.utility import print_progress, file_exists, load, save, get_file_path
from sklearn import preprocessing

from rootfile import ROOTPATH
from utility.utility import file_exists, load, save

GLOVE_DIR = ROOTPATH + "/data/GloveWordEmbeddings/"


class GloVe:
    dimensionCount = 0
    glove_file = ''
    model = {}

    def __init__(self, dimension_count):
        self.dimensionCount = dimension_count
        self.glove_file = "glove.6B." + str(dimension_count) + "d.txt"

    # Load model
    def load_glove_model(self):
        glove_model_file_name = "glove_model_" + self.glove_file
        if file_exists(glove_model_file_name):
            self.model = load(glove_model_file_name)
        else:
            self.load_word_embeddings(self.glove_file, glove_model_file_name)

    def load_word_embeddings(self, glove_file, save_name):
        print("Loading Glove word embeddings")
        with open(GLOVE_DIR + glove_file, 'r+', encoding="utf8") as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = torch.from_numpy(np.array([float(val) for val in split_line[1:]]))
                self.model[word] = embedding
            save(self.model, save_name)
            print("Done.", len(self.model), " words of loaded!")

    def get_weights_matrix(self, vocabulary):
        if file_exists("wm"):
            return load("wm")
        else:
            matrix_len = len(vocabulary)
            weights_matrix = torch.zeros((matrix_len, self.dimensionCount))
            words_found = 0

            for i, word in enumerate(vocabulary):
                try:
                    weights_matrix[i] = self.model[word]
                    words_found += 1
                except KeyError:
                    weights_matrix[i] = torch.from_numpy(np.random.normal(scale=0.6, size=(self.dimensionCount,)))
            save(weights_matrix, "wm")
            return weights_matrix

    # Check if features exist
    def get_features(self, emails, dataset):
        dataset_name = type(dataset).__name__
        feature_file_name = dataset_name + '_features'
        if file_exists(feature_file_name):
            return load(feature_file_name)
        self.load_glove_model()
        sum_vectors_array = self.sum_vectors(emails)
        features = preprocessing.scale(sum_vectors_array)
        save(features, feature_file_name)
        return features

    def sum_vectors(self, words_in_emails):
        all_vector_sum = []
        for words in words_in_emails:
            vector_sum = np.zeros(self.dimensionCount)
            for i in range(len(words)):
                if words[i] in self.model:
                    vector_sum += self.model[words[i]].numpy()
            all_vector_sum.append(vector_sum)
        return all_vector_sum
