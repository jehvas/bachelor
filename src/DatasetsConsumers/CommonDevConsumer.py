import json

import os

from sklearn.externals.joblib import delayed, Parallel

from DatasetsConsumers import AbstractDataset
from utility.utility import print_progress

'''
    File format:
    { topic: string; id: string; messages [ { id: string; author: string; date: string; file: string; } ] }
    '''


class CommonDevConsumer(AbstractDataset.AbstractDataset):
    MAILS_PATH = ""
    JSON_PATH = ""
    parsed_json = None
    words = []
    labels = []
    topicDict = {}
    counter = 0

    def commonLoad(self, json_path, mails_path, load_filtered_data):
        self.MAILS_PATH = mails_path
        self.JSON_PATH = json_path
        if load_filtered_data:
            load_check_result = super().pre_load()
            if load_check_result is not None:
                return load_check_result

        with open(self.JSON_PATH, 'r', encoding='UTF8') as f:
            parsed_json = json.loads(f.read())
            results = Parallel(n_jobs=-1)(
                delayed(self.parse_topic)(mail_json, self.get_topic_label(mail_json['topic']), len(parsed_json)) for
                mail_json in parsed_json[:1000])
            words, labels = zip(*results)
            print("Loaded", len(words), "emails and", len(labels), "labels")

            super().post_load(words, labels)
            return words, labels

    def parse_topic(self, mail_json, label, total):
        words = []
        labels = []
        for message_json in mail_json['messages']:
            tokenized_mail = self.tokenize_mail(self.MAILS_PATH + message_json['file'])
            words.append(tokenized_mail)
            labels.append(label)
        if label % 315 == 0:
            print_progress(label, total)
        return words, labels

    def tokenize_topic(self, mail_json):
        words = []
        for message_json in mail_json['messages']:
            tokenize = self.tokenize_mail(self.MAILS_PATH + message_json['file'])
            if tokenize is not None:
                words.append(tokenize)
        return words

    def tokenize_mail(self, file_path):
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='UTF8') as mail:
                mail_words = self.process_single_mail(mail.read())
                return mail_words

    def get_topic_label(self, topic):
        if topic in self.topicDict:
            return self.topicDict[topic]
        else:
            self.topicDict[topic] = self.counter
            self.counter += 1
            return self.counter - 1
