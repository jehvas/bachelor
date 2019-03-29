import abc

import time

import itertools

import numpy as np

from utility.utility import save, load, file_exists
from nltk import word_tokenize
from nltk.corpus import stopwords


class AbstractDataset(abc.ABC):
    stop_words = set(stopwords.words("english"))
    vocabulary = {}
    word_count_list = []

    @abc.abstractmethod
    def load(self, load_filtered_data=False):
        pass

    def pre_load(self):
        caller_name = type(self).__name__
        print("Being loading dataset:", caller_name)
        if file_exists(caller_name + "_saved_mails") and file_exists(caller_name + "_saved_labels"):
            emails = load(caller_name + "_saved_mails")
            labels = load(caller_name + "_saved_labels")
            self.setVocabulary(emails)
            self.finalize(caller_name, emails, labels)
            return emails, labels
        else:
            print("Saved mails and labels not found... Creating them\n")
            return None

    def post_load(self, emails, labels):
        caller_name = type(self).__name__
        if type(emails) is not np.ndarray or type(labels) is not np.ndarray:
            raise Exception("Dataset must return numpy arrays!")
        self.finalize(caller_name, emails, labels)
        self.setVocabulary(emails)
        save(emails, caller_name + "_saved_mails")
        save(labels, caller_name + "_saved_labels")

    def finalize(self, name, emails, labels):
        if len(emails) != len(labels):
            raise Exception("length of emails & labels should match!!")

        print("Finished loading dataset:", name, "\t\t", "Size: ", len(emails), ",", len(labels))

    def process_single_mail(self, text):
        text_tokenized = word_tokenize(text.lower())
        sentence_no_stop_words = self.filter_stop_words(text_tokenized)
        email_words = [w for w in sentence_no_stop_words if w.isalpha()]
        return email_words

    def setVocabulary(self, emails):
        start_time2 = time.time()
        merged = list(itertools.chain(*emails))
        idx = 0
        for word in merged:
            if word not in self.vocabulary:
                self.vocabulary[word] = idx
                idx += 1
        for mail in emails:
            mail_dict = {}
            for word in mail:
                mail_dict[word] = mail_dict.get(word, 0) + 1
            self.word_count_list.append(mail_dict)
        print("Finished generating vocabulary\n--- %s seconds ---" % (time.time() - start_time2))

    def filter_stop_words(self, text_tokenized):
        filtered_sentence = []
        for w in text_tokenized:
            if w not in self.stop_words:
                filtered_sentence.append(w)
        return filtered_sentence
