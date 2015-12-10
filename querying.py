import codecs
import string
import re
import json
import sys
from stem import IndonesianStemmer

sys.stdout = codecs.getwriter('utf8')(sys.stdout.buffer)

stopwords = list()
stopwords_text = open('stopwords_id.txt','r').readlines()
for line in stopwords_text:
    list_stopwords = stopwords.append(line.strip())

punctuation = list(string.punctuation)
stop = stopwords + punctuation + ['rt', 'via']

stemmer = IndonesianStemmer()

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def casefold(s):
    s = s.lower()
    return s

def filtering(s):
    s = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', s)
    s = re.sub('&[a-z]{1,6};', '', s)
    s = re.sub('[\`\~\!\$\%\^\&\*\(\)\-\+\=\[\]\{\}\\\|\;\:\'\"\,\<\.\>\/\?]', '', s)
    s = re.sub('(\s|^)?[0-9]+(\s|$)?', '', s)
    s = re.sub(u'[\u0000-\u001f\u007f-\uffff]', '', s)
    return s

words = list()
def stemming(s):
    for word in s:
        list_word = words.append(stemmer.stem(word))
    return words

with open('jokowi.txt', 'r') as f:
    for line in f:
        tweet = json.loads(line)
        casefolds = casefold(tweet['text'])
        clean = filtering(casefolds)
        stemmed = stemming(clean)
        tokens = preprocess(clean)
        terms_stop = [term for term in tokens if term not in stop]
        print(stemmed)