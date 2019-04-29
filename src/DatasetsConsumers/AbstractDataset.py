import abc
import time
from typing import List, Counter

import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords

from utility.utility import save, load, file_exists


def check_lengths(emails: np.ndarray, labels: np.ndarray) -> None:
    if type(labels) == tuple:
        lb1, lb2 = labels
        if len(emails) != len(lb1) and len(emails) != len(lb2):
            raise Exception("length of emails & labels should match!!")
    elif len(emails) != len(labels):
        raise Exception("length of emails & labels should match!!")


def check_types(emails: np.ndarray, labels: np.ndarray) -> None:
    if type(labels) == tuple:
        lb1, lb2 = labels
        if type(emails) is not np.ndarray or type(lb1) is not np.ndarray or type(lb2) is not np.ndarray:
            raise Exception("Dataset must return numpy arrays!")
    elif type(emails) is not np.ndarray or type(labels) is not np.ndarray:
        raise Exception("Dataset must return numpy arrays!")


class AbstractDataset(abc.ABC):
    stop_words = set(stopwords.words("english"))
    word_count_list: List = []
    classes: List = []

    @abc.abstractmethod
    def sub_load(self) -> (List[List[str]], List[int]):
        pass

    @abc.abstractmethod
    def set_classes(self) -> None:
        pass

    def load(self, load_filtered_data: bool = False) -> (List[List[str]], List[int]):
        emails, labels = None, None
        if load_filtered_data:
            load_check_result = self.pre_load()
            if load_check_result is not None:
                emails, labels = load_check_result
        if emails is None or labels is None:
            start_time = time.time()
            emails, labels = self.sub_load()
            print("--- %s seconds ---" % (time.time() - start_time))

        self.set_classes()
        # print(Counter(labels))

        emails, labels = np.asarray(emails), np.asarray(labels)
        self.post_load(emails, labels)
        return emails, labels

    def get_name(self) -> str:
        return type(self).__name__

    def pre_load(self) -> (List[List[str]], List[int]):
        caller_name = self.get_name()
        print("Being loading dataset:", caller_name)
        if file_exists(caller_name + "_saved_mails") and file_exists(caller_name + "_saved_labels"):
            emails = load(caller_name + "_saved_mails")
            labels = load(caller_name + "_saved_labels")
            check_lengths(emails, labels)
            return emails, labels
        else:
            print("Saved mails and labels not found... Creating them\n")
            return None

    def post_load(self, emails: np.ndarray, labels: np.ndarray) -> None:
        caller_name = self.get_name()
        if len(self.classes) == 0:
            self.classes = list(set(labels))

        check_types(emails, labels)
        check_lengths(emails, labels)
        print("Finished loading dataset:", caller_name, "\t\t", "Size: ", len(emails), ",", len(labels))
        save(emails, caller_name + "_saved_mails")
        save(labels, caller_name + "_saved_labels")

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
