import os
from DatasetsConsumers import AbstractDataset
from DatasetsConsumers.AbstractDataset import post_load, pre_load


class SpamHam(AbstractDataset.AbstractDataset):
    def load(self, load_filtered_data=False):
        if load_filtered_data:
            load_check_result = pre_load(self)
            if load_check_result is not None:
                return load_check_result

        direc = "../../data/emails/"
        files = os.listdir(direc)

        emails = [direc + email for email in files]
        words = []
        ec = len(emails)
        labels = []
        for email in emails:
            if "ham" in email:
                labels.append(0)
            elif "spam" in email:
                labels.append(1)
            else:
                continue  # SO .DS_Store is not processed

            if ec % 100 == 0:
                print(ec)
            ec = ec - 1
            f = open(email, encoding="latin-1")
            text = f.read()
            f.close()

            words.append(self.process_single_mail(text))
        post_load(self, words, labels)
        return words, labels
