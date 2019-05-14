from typing import List
from keras_preprocessing import sequence
from keras_preprocessing.text import Tokenizer
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from DatasetsConsumers.AbstractDataset import AbstractDataset
from rootfile import ROOTPATH
from utility.utility import file_exists, load, save
import numpy as np
import tensorflow as tf

GLOVE_DIR = ROOTPATH + "data/GloveWordEmbeddings/"


class GloVe:
    dimensionCount = 0
    glove_file = ''
    model = {}

    def __init__(self, dimension_count: int):
        self.dimensionCount = dimension_count
        self.glove_file = "glove.6B." + str(dimension_count) + "d.txt"
        # self.glove_file = 'glove.840B.300d.txt'

    # Load model
    def load_glove_model(self):
        if len(self.model) is not 0:
            return
        print('Features / Weights matrix not found.')
        print("Loading Glove word embeddings")
        num_lines = 0
        line_counter = 0
        with open(GLOVE_DIR + self.glove_file, 'r+', encoding="utf8") as f:
            for _ in f:
                num_lines += 1
        with open(GLOVE_DIR + self.glove_file, 'r+', encoding="utf8") as f:
            for line in f:
                line_counter += 1
                if line_counter % int(num_lines / 10) == 0:
                    print("{:.2f}%".format(line_counter / (int(num_lines / 100) * 100) * 100))
                split_line = line.split()
                word = split_line[0]

                if len(split_line) > self.dimensionCount + 1:
                    continue
                embedding = np.asarray(split_line[1:], dtype='float32')
                self.model[word] = embedding
            print("Done.", len(self.model), " tokens loaded!")

    def get_weights_matrix(self, emails: List[List[str]], dataset: AbstractDataset, dataset_mode):
        wm_file_name = dataset_mode + "/" + "{}_weights_matrix_{}".format(dataset.get_name(), self.dimensionCount)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(emails)
        sequences = tokenizer.texts_to_sequences(emails)
        sequences_matrix = sequence.pad_sequences(sequences, maxlen=256)
        if file_exists(wm_file_name):
            return load(wm_file_name), sequences_matrix

        self.load_glove_model()
        vocab_size = len(tokenizer.word_index) + 1
        weights_matrix = np.zeros((vocab_size, self.dimensionCount))
        for word, i in tokenizer.word_index.items():
            embedding_vector = self.model.get(word)
            if embedding_vector is not None:
                weights_matrix[i] = embedding_vector
        weights_matrix = tf.convert_to_tensor(weights_matrix)
        save(weights_matrix, wm_file_name)
        return weights_matrix, sequences_matrix

    # Check if features exist
    def get_features(self, emails: np.array, dataset: AbstractDataset):
        print("Loading embedding features")
        feature_file_name = dataset.mode + "/" + dataset.get_name() + '_features_' + str(self.dimensionCount)
        if file_exists(feature_file_name):
            return load(feature_file_name)
        self.load_glove_model()
        sum_vectors_array = self.sum_vectors(emails)
        features = preprocessing.scale(sum_vectors_array)
        save(features, feature_file_name)
        return features

    def sum_vectors(self, words_in_emails):
        all_vector_sum = []
        for i in range(len(words_in_emails)):
            words = words_in_emails[i]
            vector_sum = np.zeros(self.dimensionCount)
            for word in words:
                if word in self.model:
                    word_vector = self.model[word]
                    vector_sum += word_vector
            all_vector_sum.append(vector_sum)
        scaler = MinMaxScaler()
        scaler.fit(all_vector_sum)
        MinMaxScaler(copy=True, feature_range=(0, 1))
        normed_vectors = scaler.transform(all_vector_sum)
        return normed_vectors
