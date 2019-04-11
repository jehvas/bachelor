import os
import time

import itertools

import numpy
from joblib import Parallel, delayed
import nltk

from DatasetsConsumers.AbstractDataset import AbstractDataset
from rootfile import ROOTPATH
from utility import utility


class Newsgroups(AbstractDataset):
    label_names = []

    def load(self, load_filtered_data=False):
        direc = ROOTPATH + "data/20Newsgroups/"
        subdirecs = self.get_subdirectories(direc)
        self.classes = subdirecs

        if load_filtered_data:
            load_check_result = super().pre_load()
            if load_check_result is not None:
                return load_check_result


        emails = []
        labels = []
        start_time = time.time()
        val = Parallel(n_jobs=-1)(delayed(self.parse_email_category)(direc + i + "/") for i in subdirecs)
        for i in range(len(val)):
            labels += ([i] * len(val[i]))

        for sublist in val:
                emails = emails + sublist

        print("--- %s seconds ---" % (time.time() - start_time))
        emails, labels = numpy.asarray(emails), numpy.asarray(labels)
        super().post_load(emails, labels)
        return emails, labels

    def parse_email_category(self, path):
        print(path)
        words = []
        files = os.listdir(path)

        emails = [path + email for email in files]
        for email in emails:
            f = open(email, encoding="latin-1")
            text = f.read()
            f.close()
            words.append(self.process_single_mail(text))
        return words

    def get_subdirectories(self, path):
        subdirectories = []
        for item in os.listdir(path):
            if os.path.isdir(os.path.join(path, item)):
                subdirectories.append(item)
        return subdirectories
