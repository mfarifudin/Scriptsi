import re
import itertools
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer

class Preprocess:

    @classmethod
    def casefold(self, word):
        return word.lower()

    @classmethod
    def remove_url(self, word):
        return re.sub('((www\.[^\s]+)|(https?:[^\s]+))', ' ', word)

    @classmethod
    def remove_mention(self, word):
        return re.sub('(?:@[\w_]+)', ' ', word)

    @classmethod
    def remove_hashtag(self, word):
        return re.sub('(?:\#+[\w_]+[\w\'_\-]*[\w_]+)', ' ', word)

    @classmethod
    def remove_symbol(self, word):
        return re.sub('[\`\~\!\$\%\^\&\*\(\)\-\+\=\[\]\{\}\\\|\;\:\'\"\,\<\.\>\/\?]', ' ', word)

    @classmethod
    def remove_number(self, word):
        return re.sub('(\s|^)?[0-9]+(\s|$)?', ' ', word)

    @classmethod
    def remove_html_char(self, word):
        return re.sub('&#?[a-z0-9]{1,6};', ' ', word)

    @classmethod
    def remove_repeated(self, word):
        return ''.join(char for char, _ in itertools.groupby(word))

    @classmethod
    def remove_rt_via(self, word):
        word = re.sub('(\s^)?rt(\s$)?', ' ', word)
        word = re.sub('(\s^)?via(\s$)?', ' ', word)
        return word

    @classmethod
    def remove_nonlatin(self, word):
        return re.sub(u'[\u0000-\u001f\u007f-\uffff]', ' ', word)

    @classmethod
    def preprocessing(self, word):
        word = self.casefold(word)
        word = self.remove_url(word)
        word = self.remove_mention(word)
        word = self.remove_hashtag(word)
        word = self.remove_rt_via(word)
        word = self.remove_html_char(word)
        word = self.remove_symbol(word)
        word = self.remove_nonlatin(word)
        word = self.remove_number(word)
        word = self.remove_repeated(word)

        return word

class Tokenizer:

    @classmethod
    def tokenize(self, word):
        tokenized = word_tokenize(word, language='english')
        stemmed = [stemmer.stem(term) for term in tokenized if len(term)>2]
        return stemmed

stemmer = PorterStemmer()