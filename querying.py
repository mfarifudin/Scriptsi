import codecs
import string
import re
import sys
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import mysql.connector
from collections import Counter, defaultdict
import operator

sys.stdout = codecs.getwriter('utf8')(sys.stdout.buffer)

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

conn = mysql.connector.connect(user='root', password='', host='127.0.0.1', database='tweets')
c = conn.cursor()

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

'''def stemming(s):
    for word in s:
        words = stemmer.stem(word)
    return words'''


sql = "SELECT post FROM bpl LIMIT 40000"

com = defaultdict(lambda : defaultdict(int))
try:
    c.execute(sql)
    results = c.fetchall()
    count_all = Counter()
    for row in results:
        post = row[0]
        #print("%s" % (post))
        process = preprocess(filtering(casefold(post)))
        terms_only = [term for term in process if term not in stop and not term.startswith(('#', '@')) and len(term)>2]
        #count_all.update(terms_only)

        for i in range(len(terms_only)-1):
            for j in range(i+1, len(terms_only)):
                w1, w2 = sorted([terms_only[i], terms_only[j]])
                if w1 != w2:
                    com[w1][w2] += 1
    com_max = []
    # For each term, look for the most common co-occurrent terms
    for t1 in com:
        t1_max_terms = max(com[t1].items(), key=operator.itemgetter(1))[:5]
        for t2 in t1_max_terms:
            com_max.append(((t1, t2), com[t1][t2]))
    # Get the most frequent co-occurrences
    terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)
    print(terms_max[:5])
    #(count_all.most_common(10))
except:
   print ("Error: unable to fetch data")

# disconnect from server
conn.close()