import abc

from nltk import word_tokenize
from nltk.corpus import stopwords
from utility.utility import save, load, file_exists


class AbstractDataset(abc.ABC):
    stop_words = set(stopwords.words("english"))

    @abc.abstractmethod
    def load(self, load_filtered_data=False):
        pass

    def pre_load(self):
        caller_name = type(self).__name__
        print("Being loading dataset:", caller_name)
        if file_exists(caller_name + "_saved_mails") and file_exists(caller_name + "_saved_labels"):
            words = load(caller_name + "_saved_mails")
            labels = load(caller_name + "_saved_labels")
            return words, labels
        else:
            print("Saved mails and labels not found... Creating them\n")
            return None

    def post_load(self, emails, labels):
        caller_name = type(self).__name__
        print("Finished loading dataset:", caller_name)
        save(emails, caller_name + "_saved_mails")
        save(labels, caller_name + "_saved_labels")

    def process_single_mail(self, text):
        texttokenized = word_tokenize(text.lower())
        sentence_no_stop_words = self.filter_stop_words(texttokenized)
        email_words = [w for w in sentence_no_stop_words if w.isalpha()]
        return email_words

    def filter_stop_words(self, text_tokenized):
        filtered_sentence = []
        for w in text_tokenized:
            if w not in self.stop_words:
                filtered_sentence.append(w)
        return filtered_sentence
