import abc
from collections import Counter
from joblib import Parallel, delayed
from DatasetsConsumers.AbstractDataset import AbstractDataset
from rootfile import ROOTPATH
from utility import load_trees


def process_line(i, line):
    if i % 10000 == 0:
        print(i)
    tree = load_trees.Node(None)
    load_trees.parse_line(line, 2, tree)
    email = load_trees.output_sentence(tree)
    label = 0 if tree.syntax == "0" else 1
    return email, label


class SensitiveEnron(AbstractDataset):
    @abc.abstractmethod
    def common_load(self, data_path):
        directories = ROOTPATH + data_path
        with open(directories + "/train.txt", encoding="latin-1") as f:
            val_from_train = Parallel(n_jobs=-1)(delayed(process_line)(i, line) for i, line in enumerate(f))
        with open(directories + "/test.txt", encoding="latin-1") as f:
            val_from_test = Parallel(n_jobs=-1)(delayed(process_line)(i, line) for i, line in enumerate(f))
        with open(directories + "/dev.txt", encoding="latin-1") as f:
            val_from_dev = Parallel(n_jobs=-1)(delayed(process_line)(i, line) for i, line in enumerate(f))

        emails_from_train, labels_from_train = zip(*val_from_train)
        emails_from_test, labels_from_test = zip(*val_from_test)
        emails_from_dev, labels_from_dev = zip(*val_from_dev)

        emails = emails_from_train + emails_from_test + emails_from_dev
        labels = labels_from_train + labels_from_test + labels_from_dev
        print('Labels:', Counter(labels))
        return emails, labels
