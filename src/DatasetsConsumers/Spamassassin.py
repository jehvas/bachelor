import os
import re

from DatasetsConsumers.AbstractDataset import AbstractDataset
from utility import utility


class Spamassassin(AbstractDataset):
    def load(self, load_filtered_data=False):
        if load_filtered_data:
            load_check_result = super().pre_load()
            if load_check_result is not None:
                return load_check_result

        direcs = ["../../data/SpamAssassin/easy_ham/", "../../data/SpamAssassin/spam_2/"]

        words = []
        labels = []

        regex = r"/(<([^>]+)>)/"
        for direc in direcs:
            files = os.listdir(direc)

            emails = [direc + email for email in files]
            ec = len(emails)
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
                #text_without_html = re.sub(regex, "", text)
                f.close()

                words.append(self.process_single_mail(text))

        self.post_load(words, labels)
        return words, labels
