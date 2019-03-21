import math

import numpy as np
import os

import torch

from utility.utility import print_progress, file_exists, load, save, get_file_path

GLOVE_DIR = "../../../data/GloveWordEmbeddings/"


class GloVe:
    dimensionCount = 0
    model = {}
    features = None
    file_name = ""

    def __init__(self, glove_file, dataset):
        dataset_name = type(dataset).__name__
        self.file_name = glove_file + "_" + dataset_name
        print(self.file_name)
        self.load_glove_model(glove_file)

    def load_word_embeddings(self, glove_file):
        print("Loading Glove word embeddings")
        with open(GLOVE_DIR + glove_file, 'r+', encoding="utf8") as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = torch.from_numpy(np.array([float(val) for val in split_line[1:]]))
                self.model[word] = embedding
            print("Done.", len(self.model), " words of loaded!")
            save(self.model, self.file_name)
            self.dimensionCount = len(next(iter(self.model.values())))

    def load_glove_model(self, glove_file):
        if file_exists(self.file_name):
            self.model = load(self.file_name)
            self.dimensionCount = len(next(iter(self.model.values())))
        else:
            self.load_word_embeddings(glove_file)

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

    '''def get_features(self, emails):
        if self.features is not None:
            return self.features
        sum_vectors_array = self.sum_vectors(emails)
        rms_array = self.root_mean_square(sum_vectors_array)
        self.features = self.create_sentence_vector(rms_array, sum_vectors_array)
        save(self.features, self.file_name)
        return self.features

    def sum_vectors(self, words_in_emails):
        all_vector_sum = []
        for words in words_in_emails:
            vector_sum = np.zeros(self.dimensionCount)
            for i in range(len(words)):
                for j in range(self.dimensionCount):
                    if words[i] in self.model:
                        vector_sum[j] += self.model[words[i]][j]
            all_vector_sum.append(vector_sum)
        return all_vector_sum

    def root_mean_square(self, vectors_sum_2d_array):
        rms_array = []
        for vector_sum in vectors_sum_2d_array:
            rms = 0
            for i in range(len(vector_sum)):
                rms = rms + vector_sum[i] * vector_sum[i]
            rms = math.sqrt(rms / self.dimensionCount)
            rms_array.append(rms)
        return rms_array

    def create_sentence_vector(self, rms_array, vectors_sum_2d_array):
        sentence_vectors = vectors_sum_2d_array
        for i in range(len(vectors_sum_2d_array)):
            for j in range(len(vectors_sum_2d_array[i])):
                if rms_array[i] == 0:
                    sentence_vectors[i][j] = 0
                else:
                    sentence_vectors[i][j] = vectors_sum_2d_array[i][j] / rms_array[i]
        return sentence_vectors'''
