import os
import re

from DatasetsConsumers.AbstractDataset import AbstractDataset
from utility import utility


class Spamassassin(AbstractDataset):
    def load(self, load_filtered_data=False):
        if load_filtered_data:
            if (os.path.exists(utility.output_path + "Spamassassin_saved_mails")) \
                    and (os.path.exists(utility.output_path + "Spamassassin_saved_labels")):
                words = utility.load(utility.output_path + "Spamassassin_saved_mails")
                labels = utility.load(utility.output_path + "Spamassassin_saved_labels")
                return words, labels
            else:
                print("Saved mails and labels not found... Creating them\n")

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

        utility.save(words, utility.output_path + "Spamassassin_saved_mails")
        utility.save(labels, utility.output_path + "Spamassassin_saved_labels")
        return words, labels
