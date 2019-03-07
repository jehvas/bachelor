import abc

from nltk import word_tokenize
from nltk.corpus import stopwords

from utility import utility


class AbstractDataset(abc.ABC):
    stop_words = set(stopwords.words("english"))

    @abc.abstractmethod
    def load(self, load_filtered_data=False):
        pass

    def process_single_mail(self, text):
        texttokenized = word_tokenize(text.lower())
        sentence_no_stop_words = self.filter_stop_words(texttokenized)
        email_words = [w for w in sentence_no_stop_words if w.isalpha()]
        return email_words

    def filter_stop_words(self, texttokenized):
        filtered_sentence = []
        for w in texttokenized:
            if w not in self.stop_words:
                filtered_sentence.append(w)
        return filtered_sentence
