import os

from DatasetsConsumers.AbstractDataset import AbstractDataset
from rootfile import ROOTPATH


class SpamHam(AbstractDataset):
    def set_classes(self) -> None:
        self.classes = ['Ham', 'Spam']

    def sub_load(self):

        directory = ROOTPATH + "data/emails/"
        files = os.listdir(directory)

        emails = [directory + email for email in files]
        words = []
        ec = 0
        labels = []
        for email in emails:
            if "ham" in email:
                labels.append(0)
            elif "spam" in email:
                labels.append(1)
            else:
                continue  # So .DS_Store is not processed

            if ec % 1000 == 0:
                print('Loaded', ec, 'of', len(emails))
            ec += 1
            f = open(email, encoding="latin-1")
            text = f.read()
            f.close()

            words.append(self.process_single_mail(text))
        return words, labels
