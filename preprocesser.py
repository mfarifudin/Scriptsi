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
        return re.sub('((www\.[^\s]+)|(https?:[^\s]+))', '', word)

    @classmethod
    def remove_mention(self, word):
        return re.sub('(?:@[\w_]+)', '', word)

    @classmethod
    def remove_hashtag(self, word):
        return re.sub('(?:\#+[\w_]+[\w\'_\-]*[\w_]+)', '', word)

    @classmethod
    def remove_rt_via(self, word):
        word = re.sub('(\s^)?rt(\s$)?', '', word)
        word = re.sub('(\s^)?via(\s$)?', '', word)
        return word

    @classmethod
    def remove_symbol(self, word):
        return re.sub('[\`\~\!\$\%\^\&\*\(\)\-\+\=\[\]\{\}\\\|\;\:\'\"\,\<\.\>\/\?]', '', word)

    @classmethod
    def remove_number(self, word):
        return re.sub('(\s|^)?[0-9]+(\s|$)?', '', word)

    @classmethod
    def remove_html_char(self, word):
        return re.sub('&#?[a-z0-9]{1,6};', '', word)

    @classmethod
    def remove_repeated(self, word):
        return ''.join(char for char, _ in itertools.groupby(word))

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
        # word = self.remove_repeated(word)
        return word

    @classmethod
    def tokenize_and_stem(self, word):
        word = self.tokenization(word)
        word = self.remove_stopword(word)
        word = self.stemming(word)
        return word

    @classmethod
    def tokenization(self, word):
        return word_tokenize(word, language='english')

    @classmethod
    def remove_stopword(self, word):
        return [term for term in word if term not in stop_words and len(term)>2]

    @classmethod
    def stemming(self, word):
        return [stemmer.stem(term) for term in word]

stemmer = PorterStemmer()

stop_words = ["a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fify", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"]