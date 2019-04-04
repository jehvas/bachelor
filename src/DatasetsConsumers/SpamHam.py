import os

import numpy as np

from DatasetsConsumers.AbstractDataset import AbstractDataset
from rootfile import ROOTPATH


class SpamHam(AbstractDataset):
    def load(self, load_filtered_data=False) -> (np.ndarray, np.ndarray):
        if load_filtered_data:
            load_check_result = super().pre_load()
            if load_check_result is not None:
                return load_check_result

        direc = ROOTPATH + "/data/emails/"
        files = os.listdir(direc)

        emails = [direc + email for email in files]
        words = []
        ec = 0
        labels = []
        for email in emails:
            if "ham" in email:
                labels.append(0)
            elif "spam" in email:
                labels.append(1)
            else:
                continue  # SO .DS_Store is not processed

            if ec % 1000 == 0:
                print('Loaded', ec, 'of', len(emails))
            ec += 1
            f = open(email, encoding="latin-1")
            text = f.read()
            f.close()

            words.append(self.process_single_mail(text))
        words, labels = np.asarray(words), np.asarray(labels)
        super().post_load(words, labels)
        return words, labels
