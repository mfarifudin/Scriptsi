import mysql.connector
import string
import re
import time
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk import pos_tag
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

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
    s = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', s)
    s = re.sub('&#?[a-z0-9]{1,6};', ' ', s)
    s = re.sub('[\`\~\!\$\%\^\&\*\(\)\-\+\=\[\]\{\}\\\|\;\:\'\"\,\<\.\>\/\?]', ' ', s)
    s = re.sub('(\s|^)?[0-9]+(\s|$)?', ' ', s)
    s = re.sub(u'[\u0000-\u001f\u007f-\uffff]', ' ', s)
    return s

all_words = []
documents = []
tests = []
train_data = []
train_label = []
test_data = []
test_label = []

c.execute(sql)
results= c.fetchmany(size=200)
results_train = c.fetchall()
for row in results_train:
    fileids = row[0]
    post = row[1]
    labels = row[2]
    process = preprocess(filtering(post.lower()))
    terms_only = [term for term in process if term not in stop and len(term)>2]
    terms_tagged = pos_tag([stemmer.stem(word) for word in terms_only])
    #print (terms_tagged)
    documents.append((terms_only, labels))
    train_data.append(post)
    train_label.append(labels)
    for w in terms_only:
        all_words.append(w)

for row in results:
    fileids = row[0]
    post = row[1]
    labels = row[2]
    process = preprocess(filtering(post.lower()))
    terms_only = [term for term in process if term not in stop and len(term)>2]
    #terms_tagged = pos_tag([stemmer.stem(word) for word in terms_only])
    #print (terms_tagged)
    tests.append((terms_only, labels))
    test_data.append(post)
    test_label.append(labels)

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:1000]
#(all_words.most_common(15))

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

feature_sets = [(find_features(rev), category) for (rev, category) in documents]
feature_sets_test = [(find_features(rev), category) for (rev, category) in tests]
training_set = feature_sets
testing_set = feature_sets_test

'''
classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)


NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
'''
vectorizer = TfidfVectorizer(min_df=5,
                             max_df = 0.8,
                             sublinear_tf=True,
                             use_idf=True)
train_vectors = vectorizer.fit_transform(train_data)
test_vectors = vectorizer.transform(test_data)

classifier_rbf = LinearSVC()
t0 = time.time()
classifier_rbf.fit(train_vectors, train_label)
t1 = time.time()
prediction_rbf = classifier_rbf.predict(test_vectors)
t2 = time.time()
time_rbf_train = t1-t0
time_rbf_predict = t2-t1

print("Results for SVC(kernel=rbf)")
print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
print(classification_report(test_label, prediction_rbf))
print(train_data)


conn.close()