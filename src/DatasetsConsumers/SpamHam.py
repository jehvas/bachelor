import os

import AbstractDataset


class SpamHam(AbstractDataset.AbstractDataset):
    def load(self):
        direc = "emails/"
        files = os.listdir(direc)

        emails = [direc + email for email in files]
        words = []
        ec = len(emails)
        labels = []
        for email in emails:
            if (ec % 100 == 0):
                print(ec)
            ec = ec - 1
            f = open(email, encoding="latin-1")
            text = f.read()
            f.close()

            words.append(self.process_single_mail(text))

            if "ham" in email:
                labels.append(0)
            if "spam" in email:
                labels.append(1)
        return words, labels