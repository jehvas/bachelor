import os
from typing import List

from DatasetsConsumers.AbstractDataset import AbstractDataset
from rootfile import ROOTPATH


class Spamassassin(AbstractDataset):
    def sub_load(self):
        direcs = [ROOTPATH + "data/SpamAssassin/easy_ham/", ROOTPATH + "data/SpamAssassin/spam_2/"]

        words: List[List[str]] = []
        labels: List[int] = []

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
        return words, labels

    def set_classes(self):
        self.classes = ['Ham', 'Spam']