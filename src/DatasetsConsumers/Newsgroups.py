import os
import time
from joblib import Parallel, delayed

from DatasetsConsumers.AbstractDataset import AbstractDataset
from utility import utility


class Newsgroups(AbstractDataset):
    def load(self, load_filtered_data=False):
        if load_filtered_data:
            load_check_result = super().pre_load()
            if load_check_result is not None:
                return load_check_result

        direc = "../../data/20Newsgroups/"
        subdirecs = self.get_subdirectories(direc)

        words = []
        labels = []
        start_time = time.time()
        val = Parallel(n_jobs=-1)(delayed(self.test)(direc + i + "/") for i in subdirecs)
        for i in range(len(val)):
            labels += ([i] * len(val[i]))

        for sublist in val:
            for item in sublist:
                words.append(item)

        print("--- %s seconds ---" % (time.time() - start_time))
        super().post_load(words, labels)
        return words, labels

    def test(self, path):
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
