import mysql.connector
import string
import re
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk import pos_tag
import nltk

conn = mysql.connector.connect(user='root', password='', host='127.0.0.1', database='tweets')
c = conn.cursor()

sql = "SELECT id, post, label FROM bpl_train LIMIT 30"

stop_words = list(stopwords.words('english'))

punctuation = list(string.punctuation)
stop = stop_words + punctuation + ['rt', 'via']

stemmer = PorterStemmer()

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>',
    r'(?:@[\w_]+)',
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',
    r"(?:[a-z][a-z'\-_]+[a-z])",
    r'(?:[\w_]+)',
    r'(?:\S)'
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

def filtering(s):
    s = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', s)
    s = re.sub('&[a-z]{1,6};', '', s)
    s = re.sub('[\`\~\!\$\%\^\&\*\(\)\-\+\=\[\]\{\}\\\|\;\:\'\"\,\<\.\>\/\?]', '', s)
    s = re.sub('(\s|^)?[0-9]+(\s|$)?', '', s)
    s = re.sub(u'[\u0000-\u001f\u007f-\uffff]', '', s)
    return s

all_words = []
documents = []

c.execute(sql)
results = c.fetchall()
for row in results:
    fileids = row[0]
    post = row[1]
    labels = row[2]
    process = preprocess(filtering(post.lower()))
    terms_only = [term for term in process if term not in stop and len(term)>2]
    terms_tagged = pos_tag([stemmer.stem(word) for word in terms_only])
    #print (terms_tagged)
    documents.append((terms_only, labels))
    for w in terms_only:
        all_words.append(w)

all_words = nltk.FreqDist(all_words)
#word_features = list(all_words.keys())[:1000]
print(all_words.most_common(15))
'''
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

print((find_features(terms_tagged)))
featuresets = [(find_features(rev), category) for (rev, category) in terms_tagged]

training_set = featuresets[:100]
testing_set = featuresets[100:]
'''
conn.close()