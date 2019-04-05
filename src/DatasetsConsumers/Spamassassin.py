import os
from typing import List

import numpy as np

from DatasetsConsumers.AbstractDataset import AbstractDataset
from rootfile import ROOTPATH


class Spamassassin(AbstractDataset):
    def load(self, load_filtered_data: bool=False) -> (List[List[str]], List[int]):
        self.classes = ['Ham', 'Spam']

        if load_filtered_data:
            load_check_result = super().pre_load()
            if load_check_result is not None:
                return load_check_result

        direcs = [ROOTPATH + "/data/SpamAssassin/easy_ham/", ROOTPATH + "/data/SpamAssassin/spam_2/"]

        words: List[List[str]] = []
        labels: List[int] = []

        regex = r"/(<([^>]+)>)/"
        for direc in direcs:
            files = os.listdir(direc)

            emails: List[str] = [direc + email for email in files]
            ec: int = len(emails)
            for email in emails:
                if "ham" in direc:
                    labels.append(0)
                elif "spam" in direc:
                    labels.append(1)
                else:
                    continue  # SO .DS_Store is not processed

                if ec % 100 == 0:
                    print(ec, " ", direc)
                ec = ec - 1
                f = open(email, encoding="latin-1")
                text = f.read()
                # text_without_html = re.sub(regex, "", text)
                f.close()

                words.append(self.process_single_mail(text))
        words = np.array(words)
        labels = np.array(labels)
        self.post_load(words, labels)
        return words, labels
