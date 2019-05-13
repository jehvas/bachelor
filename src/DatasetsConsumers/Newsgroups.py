import os

from joblib import Parallel, delayed
from DatasetsConsumers.AbstractDataset import AbstractDataset
from rootfile import ROOTPATH


def get_subdirectories(path):
    subdirectories = []
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path, item)):
            subdirectories.append(item)
    return subdirectories


class Newsgroups(AbstractDataset):
    def sub_load(self):
        directories = ROOTPATH + "data/20Newsgroups/"
        sub_directories = get_subdirectories(directories)

        emails = []
        labels = []
        val = Parallel(n_jobs=-1)(delayed(self.parse_email_category)(directories + i + "/") for i in sub_directories)
        for i in range(len(val)):
            labels += ([i] * len(val[i]))

        for sublist in val:
            emails = emails + sublist
        return emails, labels

    def set_classes(self):
        self.classes = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                        'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
                        'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
                        'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast',
                        'talk.politics.misc', 'talk.religion.misc']

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
