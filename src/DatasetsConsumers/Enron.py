import csv
import email
import os
import sys

from DatasetsConsumers.AbstractDataset import AbstractDataset
from utility.utility import print_progress

ENRON_FILE_PATH = "../../data/Enron/emails.csv"
ENRON_ROW_COUNT = 517401
csv.field_size_limit(1_000_000)


class Enron(AbstractDataset):
    sender_dict = {}
    counter = 0

    def load(self, load_filtered_data=False):
        if load_filtered_data:
            load_check_result = super().pre_load()
            if load_check_result is not None:
                return load_check_result

        with open(ENRON_FILE_PATH, mode='r', encoding='UTF8') as f:
            email_words = []
            labels = []
            reader = csv.reader(f)
            next(reader, None)  # skip the headers
            progress = 0
            for row in reader:
                email_object = email.message_from_string(row[1])
                mail_content = email_object.get_payload()
                mail_words = self.process_single_mail(mail_content)
                email_words.append(mail_words)

                sender = email_object['From']
                sender_label = self.get_sender_label(sender)
                labels.append(sender_label)

                if progress % int(ENRON_ROW_COUNT / 100) == 0:
                    print_progress(progress, ENRON_ROW_COUNT)
                    if progress > 7_401:
                        break
                progress += 1

        super().post_load(email_words, labels)
        return email_words, labels

    def tokenize_mail(self, file_path):
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='UTF8') as mail:
                mail_words = self.process_single_mail(mail.read())
                return mail_words

    def get_sender_label(self, name):
        if name in self.sender_dict:
            return self.sender_dict[name]
        else:
            self.sender_dict[name] = self.counter
            self.counter += 1
            return self.counter - 1
