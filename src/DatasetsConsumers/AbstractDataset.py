import abc

from nltk import word_tokenize
from nltk.corpus import stopwords


class AbstractDataset(abc.ABC):
    @abc.abstractmethod
    def load(self):
        pass

    def process_single_mail(self, text):
        texttokenized = word_tokenize(text.lower())
        sentence_no_stop_words = self.filter_stop_words(texttokenized)
        email_words = [w for w in sentence_no_stop_words if w.isalpha()]
        return email_words

    stop_words = set(stopwords.words("english"))

    def filter_stop_words(self, texttokenized):
        filtered_sentence = []
        for w in texttokenized:
            if w not in self.stop_words:
                filtered_sentence.append(w)
        return filtered_sentence
