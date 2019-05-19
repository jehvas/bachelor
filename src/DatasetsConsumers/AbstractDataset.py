import abc
import time
import numpy as np

from typing import List
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

from utility.undersample_split import resize_under_sample
from utility.utility import save, file_exists, load


def check_lengths(emails, labels):
    if len(emails) != len(labels):
        raise Exception("length of emails & labels should match!!")


def check_types(emails, labels):
    if type(labels) == tuple:
        lb1, lb2 = labels
        if type(emails) is not np.ndarray or type(lb1) is not np.ndarray or type(lb2) is not np.ndarray:
            raise Exception("Dataset must return numpy arrays!")
    elif type(emails) is not np.ndarray or type(labels) is not np.ndarray:
        raise Exception("Dataset must return numpy arrays!")


class AbstractDataset(abc.ABC):
    stop_words = set(stopwords.words("english"))
    classes = []
    mode = None

    @abc.abstractmethod
    def sub_load(self):
        pass

    @abc.abstractmethod
    def set_classes(self):
        pass

    def load(self, load_filtered_data=True, dataset_mode='standard'):
        emails, labels = None, None
        self.mode = dataset_mode
        if load_filtered_data:
            load_check_result = self.pre_load()
            if load_check_result is not None:
                emails, labels = load_check_result
        if emails is None or labels is None:
            start_time = time.time()
            emails, labels = self.sub_load()
            emails = [self.process_single_mail(email) for email in emails]
            save(emails, self.get_name() + "_saved_mails")
            save(labels, self.get_name() + "_saved_labels")
            print("--- %s seconds ---" % (time.time() - start_time))

        self.set_classes()

        emails, labels = np.asarray(emails), np.asarray(labels)
        self.post_load(emails, labels)
        if dataset_mode == "2000":
            emails, labels = resize_under_sample(emails, labels)

        return emails, labels

    def get_name(self):
        return type(self).__name__

    def pre_load(self):
        print("Being loading dataset:", self.get_name())
        if file_exists(self.get_name() + "_saved_mails") and file_exists(self.get_name() + "_saved_labels"):
            emails = load(self.get_name() + "_saved_mails")
            labels = load(self.get_name() + "_saved_labels")
            check_lengths(emails, labels)
            return emails, labels
        else:
            print("Saved mails and labels not found... Creating them\n")
            return None

    def post_load(self, emails: np.ndarray, labels: np.ndarray):
        if len(self.classes) == 0:
            self.classes = list(set(labels))

        check_types(emails, labels)
        check_lengths(emails, labels)
        print("Finished loading dataset:", self.get_name(), "\t\t", "Size: ", len(emails), ",", len(labels))

    def process_single_mail(self, text):
        text_tokenized = word_tokenize(text.lower())
        sentence_no_stop_words = self.filter_stop_words(text_tokenized)
        email_words = [w for w in sentence_no_stop_words if w.isalpha()]
        return email_words

    def filter_stop_words(self, text_tokenized):
        filtered_sentence = []
        for w in text_tokenized:
            if w not in self.stop_words:
                filtered_sentence.append(w)
        return filtered_sentence
