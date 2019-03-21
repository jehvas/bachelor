import email
import os

from joblib import Parallel, delayed
'''
    File format:
    { topic: string; id: string; messages [ { id: string; author: string; date: string; file: string; } ] }
    '''
import abc

from DatasetsConsumers.AbstractDataset import AbstractDataset


class CommonDevConsumer(AbstractDataset):
    MAILS_PATH = ""
    CUT_OFF_VALUE = 20
    topicDict = {}
    counter = 0

    @abc.abstractmethod
    def common_load(self, json_path, mails_path, load_filtered_data):
        self.MAILS_PATH = mails_path
        with open(json_path, 'r', encoding='UTF8') as f:
            import json
            parsed_json = json.loads(f.read())
            results = Parallel(n_jobs=-1)(
                delayed(self.parse_topic)(mail_json, self.get_topic_label(mail_json['topic']), len(parsed_json)) for
                mail_json in parsed_json)
            tmp_emails, tmp_labels = zip(*results)

            emails = []
            for l in tmp_emails:
                emails = emails + l
            labels = []
            for l in tmp_labels:
                labels = labels + l
            return emails, labels

    def parse_topic(self, mail_json, label, total):
        emails = []
        labels = []
        if len(mail_json['messages']) >= self.CUT_OFF_VALUE:
            for message_json in mail_json['messages']:
                tokenized_mail = self.tokenize_mail(self.MAILS_PATH + message_json['file'])
                if tokenized_mail is not None:
                    emails.append(tokenized_mail)
                    labels.append(label)
            if label % 315 == 0:
                from utility.utility import print_progress
                print_progress(label, total)
        return emails, labels

    def tokenize_mail(self, file_path):
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='UTF8') as mail:
                email_object = email.message_from_string(mail.read())
                mail_content = ""
                if email_object.is_multipart():
                    for payload in email_object.get_payload():
                        mail_content += str(payload)
                else:
                    mail_content = email_object.get_payload()
                mail_words = self.process_single_mail(mail_content)
                return mail_words

    def get_topic_label(self, topic):
        if topic in self.topicDict:
            return self.topicDict[topic]
        else:
            self.topicDict[topic] = self.counter
            self.counter += 1
            return self.counter - 1

    def n_categories(self):
        return self.counter
