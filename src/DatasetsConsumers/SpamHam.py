import os
from DatasetsConsumers import AbstractDataset

from utility import utility


class SpamHam(AbstractDataset.AbstractDataset):
    def load(self, load_filtered_data=False):
        if load_filtered_data:
            if (os.path.exists(utility.output_path + "SpamHam_saved_mails")) \
                    and (os.path.exists(utility.output_path + "SpamHam_saved_labels")):
                words = utility.load(utility.output_path + "SpamHam_saved_mails")
                labels = utility.load(utility.output_path + "SpamHam_saved_labels")
                return words, labels
            else:
                print("Saved mails and labels not found... Creating them\n")

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

        utility.save(words, utility.output_path + "SpamHam_saved_mails")
        utility.save(labels, utility.output_path + "SpamHam_saved_labels")
        return words, labels
