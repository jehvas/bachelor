import math

import numpy as np
import os
from utility.utility import print_progress, file_exists, load, save

GLOVE_DIR = "../../data/GloveWordEmbeddings/"


class GloVe:
    dimensionCount = 0
    model = {}

    def __init__(self, glove_file):
        self.load_glove_model(glove_file)

    def load_glove_model(self, glove_file):
        if file_exists("Glove_saved_gModel"):
            self.model = load("Glove_saved_gModel")
            self.dimensionCount = len(next(iter(self.model.values())))
        else:
            print("Loading Glove Model")
            with open(GLOVE_DIR + glove_file, 'r+', encoding="utf8") as f:
                # total of 1917494 lines in glove.42B.300d.txt
                total = os.stat(glove_file).st_size
                c = 0
                for line in f:
                    split_line = line.split()
                    word = split_line[0]
                    embedding = np.array([float(val) for val in split_line[1:]])
                    self.model[word] = embedding
                    if c % 50000 == 0:
                        print_progress(c, total)
                    c = c + len(line)
                print("c: ", c, "total: ", total)
                print("Done.", len(self.model), " words of loaded!")
                save(self.model, "Glove_saved_gModel")
                self.dimensionCount = len(next(iter(self.model.values())))

    def get_features(self, emails):
        sum_vectors_array = self.sum_vectors(emails)
        rms_array = self.root_mean_square(sum_vectors_array)
        return self.create_sentence_vector(rms_array, sum_vectors_array)

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
        return sentence_vectors
