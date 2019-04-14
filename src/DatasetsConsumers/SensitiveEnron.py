import abc
import time
from collections import Counter

import numpy
from joblib import Parallel, delayed

from DatasetsConsumers.AbstractDataset import AbstractDataset
from rootfile import ROOTPATH
from utility import load_trees


class SensitiveEnron(AbstractDataset):
    label_names = []

    @abc.abstractmethod
    def common_load(self, data_path, load_filtered_data):
        direc = ROOTPATH + data_path

        start_time = time.time()

        with open(direc + "/train.txt", encoding="latin-1") as f:
            val_from_train = Parallel(n_jobs=-1)(delayed(self.process_line)(i, line) for i, line in enumerate(f))
        with open(direc + "/test.txt", encoding="latin-1") as f:
            val_from_test = Parallel(n_jobs=-1)(delayed(self.process_line)(i, line) for i, line in enumerate(f))
        with open(direc + "/dev.txt", encoding="latin-1") as f:
            val_from_dev = Parallel(n_jobs=-1)(delayed(self.process_line)(i, line) for i, line in enumerate(f))

        emails_from_train, labels_from_train = zip(*val_from_train)
        emails_from_test, labels_from_test = zip(*val_from_test)
        emails_from_dev, labels_from_dev = zip(*val_from_dev)

        emails = emails_from_train + emails_from_test + emails_from_dev
        labels = labels_from_train + labels_from_test + labels_from_dev

        print("--- %s seconds ---" % (time.time() - start_time))
        emails, labels = numpy.asarray(emails), numpy.asarray(labels)
        print(Counter(labels))
        return emails, labels

    def process_line(self, i, line):
        if i % 10000 == 0:
            print(i)
        tree = load_trees.Node(None)
        load_trees.parse_line(line, 2, tree)
        email = load_trees.output_sentence(tree)
        label = 0 if tree.syntax == "0" else 1
        return email, label
