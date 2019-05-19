import os
from DatasetsConsumers.AbstractDataset import AbstractDataset
from rootfile import ROOTPATH
from utility.utility import print_progress


class Spamassassin(AbstractDataset):
    def sub_load(self):
        directories = [ROOTPATH + "data/SpamAssassin/easy_ham/", ROOTPATH + "data/SpamAssassin/spam_2/"]

        words = []
        labels = []
        total_mails = 0
        progress = 0
        for directory in directories:
            total_mails += len(os.listdir(directory))

        for directory in directories:
            files = os.listdir(directory)
            emails = [directory + email_file for email_file in files]
            if "ham" in directory:
                mail_type = 0
            elif "spam" in directory:
                mail_type = 1
            else:
                continue
            labels = labels + [mail_type] * len(emails)
            for email in emails:
                print_progress(progress, total_mails)
                progress += 1
                with open(email, encoding="latin-1") as f:
                    words.append(f.read())
        return words, labels

    def set_classes(self):
        self.classes = ['Ham', 'Spam']
