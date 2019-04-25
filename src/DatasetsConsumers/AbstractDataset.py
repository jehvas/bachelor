import abc

import time

import itertools
from typing import List, Dict

import numpy as np

from utility.utility import save, load, file_exists
from nltk import word_tokenize
from nltk.corpus import stopwords


class AbstractDataset(abc.ABC):
    stop_words = set(stopwords.words("english"))
    word_count_list: List = []
    classes: List = []

    @abc.abstractmethod
    def load(self, load_filtered_data: bool = False) -> (List[List[str]], List[int]):
        print('abstr')
        pass

    def get_name(self) -> str:
        return type(self).__name__

    def pre_load(self) -> (List[List[str]], List[int]):
        caller_name = type(self).__name__
        print("Being loading dataset:", caller_name)
        if file_exists(caller_name + "_saved_mails") and file_exists(caller_name + "_saved_labels"):
            emails = load(caller_name + "_saved_mails")
            labels = load(caller_name + "_saved_labels")
            self.check_lengths(emails, labels)
            return emails, labels
        else:
            print("Saved mails and labels not found... Creating them\n")
            return None

    def post_load(self, emails: np.ndarray, labels: np.ndarray) -> None:
        caller_name = type(self).__name__
        if len(self.classes) == 0:
            self.classes = list(set(labels))

        self.check_types(emails, labels)
        self.check_lengths(emails, labels)
        print("Finished loading dataset:", caller_name, "\t\t", "Size: ", len(emails), ",", len(labels))
        save(emails, caller_name + "_saved_mails")
        save(labels, caller_name + "_saved_labels")

    def check_types(self, emails: np.ndarray, labels: np.ndarray) -> None:
        if type(labels) == tuple:
            lb1, lb2 = labels
            if type(emails) is not np.ndarray or type(lb1) is not np.ndarray or type(lb2) is not np.ndarray:
                raise Exception("Dataset must return numpy arrays!")
        elif type(emails) is not np.ndarray or type(labels) is not np.ndarray:
            raise Exception("Dataset must return numpy arrays!")

    def check_lengths(self, emails: np.ndarray, labels: np.ndarray) -> None:
        if type(labels) == tuple:
            lb1, lb2 = labels
            if len(emails) != len(lb1) and len(emails) != len(lb2):
                raise Exception("length of emails & labels should match!!")
        elif len(emails) != len(labels):
            raise Exception("length of emails & labels should match!!")

    def process_single_mail(self, text: str) -> List[str]:
        text_tokenized = word_tokenize(text.lower())
        sentence_no_stop_words = self.filter_stop_words(text_tokenized)
        email_words = [w for w in sentence_no_stop_words if w.isalpha()]
        return email_words

    def filter_stop_words(self, text_tokenized: List[str]) -> List[str]:
        filtered_sentence = []
        for w in text_tokenized:
            if w not in self.stop_words:
                filtered_sentence.append(w)
        return filtered_sentence
