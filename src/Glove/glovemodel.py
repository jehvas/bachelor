import numpy as np
from sklearn import preprocessing

from rootfile import ROOTPATH
from utility.utility import file_exists, load, save

GLOVE_DIR = ROOTPATH + "/data/GloveWordEmbeddings/"


class GloVe:
    dimensionCount = 0
    model = {}
    features = None

    def __init__(self, glove_file):
        self.load_glove_model(glove_file)

    def load_glove_model(self, glove_file):
        if file_exists("Glove_saved_gModel"):
            self.features = load("Glove_saved_gModel")
        else:
            print("Loading Glove Model")
            with open(GLOVE_DIR + glove_file, 'r+', encoding="utf8") as f:
                for line in f:
                    split_line = line.split()
                    word = split_line[0]
                    embedding = np.array([float(val) for val in split_line[1:]])
                    self.model[word] = embedding
                print("Done.", len(self.model), " words of loaded!")
                self.dimensionCount = len(next(iter(self.model.values())))

    def get_features(self, emails):
        if self.features is not None:
            return self.features
        sum_vectors_array = self.sum_vectors(emails)
        features = preprocessing.scale(sum_vectors_array)
        save(self.features, "Glove_saved_gModel")
        return features

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
