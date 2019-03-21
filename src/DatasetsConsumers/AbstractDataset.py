import abc

import time

import itertools

from utility.utility import save, load, file_exists
from nltk import word_tokenize
from nltk.corpus import stopwords


class AbstractDataset(abc.ABC):
    stop_words = set(stopwords.words("english"))
    vocabulary = {}

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
        self.finalize(caller_name, emails, labels)
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

    def setVocabulary(self, words):
        start_time2 = time.time()
        merged = list(itertools.chain(*words))
        idx = 0
        for word in merged:
            if word not in self.vocabulary:
                self.vocabulary[word] = idx
                idx += 1
        print("Vocab --- %s seconds ---" % (time.time() - start_time2))

    def filter_stop_words(self, text_tokenized):
        filtered_sentence = []
        for w in text_tokenized:
            if w not in self.stop_words:
                filtered_sentence.append(w)
        return filtered_sentence
