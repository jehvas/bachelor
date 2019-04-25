from typing import Dict, List

import numpy as np
from keras_preprocessing import sequence
from keras_preprocessing.text import Tokenizer
from sklearn import preprocessing

from DatasetsConsumers.AbstractDataset import AbstractDataset
from rootfile import ROOTPATH
from utility.utility import file_exists, load, save

GLOVE_DIR = ROOTPATH + "data/GloveWordEmbeddings/"


class GloVe:
    dimensionCount = 0
    glove_file = ''
    model = {}

    def __init__(self, dimension_count: int) -> None:
        self.dimensionCount = dimension_count
        self.glove_file = "glove.6B." + str(dimension_count) + "d.txt"

    # Load model
    def load_glove_model(self) -> None:
        glove_model_file_name = "glove_model_" + self.glove_file
        if file_exists(glove_model_file_name):
            self.model = load(glove_model_file_name)
        else:
            self.load_word_embeddings(self.glove_file, glove_model_file_name)

    def load_word_embeddings(self, glove_file: str, save_name: str) -> None:
        print("Loading Glove word embeddings")
        with open(GLOVE_DIR + glove_file, 'r+', encoding="utf8") as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array([float(val) for val in split_line[1:]])
                self.model[word] = embedding
            save(self.model, save_name)
            print("Done.", len(self.model), " words of loaded!")

    def get_weights_matrix(self, emails: List[List[str]], dataset: AbstractDataset) -> np.array:
        wm_file_name = "weights_matrix_{}_{}".format(dataset.get_name(), self.dimensionCount)

        tokenizer = Tokenizer(num_words=100000)
        tokenizer.fit_on_texts(emails)
        sequences = tokenizer.texts_to_sequences(emails)
        sequences_matrix = sequence.pad_sequences(sequences, maxlen=256)
        if file_exists(wm_file_name):
            return load(wm_file_name), sequences_matrix
        else:
            if len(self.model) is 0:
                self.load_glove_model()
            weights_matrix = np.zeros((tokenizer.num_words, self.dimensionCount))
            for word, i in tokenizer.word_index.items():
                try:
                    embedding_vector = self.model.get(word)
                    if embedding_vector is not None:
                        weights_matrix[i] = embedding_vector
                except KeyError:
                    weights_matrix[i] = np.random.normal(scale=0.6, size=(self.dimensionCount,))
            save(weights_matrix, wm_file_name)
            return weights_matrix, sequences_matrix

    # Check if features exist
    def get_features(self, emails: np.array, dataset: AbstractDataset) -> np.array:
        print("Loading embedding features")
        feature_file_name = dataset.get_name() + '_features_' + str(self.dimensionCount)
        if file_exists(feature_file_name):
            return load(feature_file_name)
        self.load_glove_model()
        sum_vectors_array = self.sum_vectors(emails)
        features = preprocessing.scale(sum_vectors_array)
        save(features, feature_file_name)
        return features

    def sum_vectors2(self, words_in_emails, tfidf):
        all_vector_sum = []
        for i in range(len(words_in_emails)):
            words = words_in_emails[i]
            vector_sum = np.zeros(self.dimensionCount)
            for word in words:
                if word in self.model:
                    word_vector = self.model[word].numpy()
                    word_vector *= tfidf[i][word]
                    vector_sum += word_vector
            all_vector_sum.append(vector_sum)
        return all_vector_sum

    def sum_vectors(self, words_in_emails):
        all_vector_sum = []
        for i in range(len(words_in_emails)):
            words = words_in_emails[i]
            vector_sum = np.zeros(self.dimensionCount)
            for word in words:
                if word in self.model:
                    word_vector = self.model[word].numpy()
                    vector_sum += word_vector
            vector_sum = vector_sum / len(words)
            all_vector_sum.append(vector_sum)
        return all_vector_sum
