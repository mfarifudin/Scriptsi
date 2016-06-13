import mysql.connector
import string
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from preprocesser import *

conn = mysql.connector.connect(user='root', password='', host='127.0.0.1', database='tweets')
c = conn.cursor()

sql = "SELECT post, label FROM bpl_train LIMIT 500"

stop_words = list(stopwords.words('english'))

punctuation = list(string.punctuation)
stop = stop_words + punctuation + ['rt', 'via']

tokenizer = Tokenizer.tokenize_and_stem
preprocesser = Preprocess.preprocessing

all_words = []
documents = []
tests = []
train_data = []
train_label = []
test_data = []
test_label = []

c.execute(sql)
results= c.fetchmany(size=100)
results_train = c.fetchall()
for row in results_train:
    post = row[0]
    labels = row[1]
    # process = word_tokenize(Preprocess.preprocess(post))
    # terms_only = [term for term in process if term not in stop and len(term)>2]
    # terms_tagged = pos_tag([stemmer.stem(word) for word in terms_only])
    # print(process)
    # documents.append((terms_only, labels))
    train_data.append(post)
    train_label.append(labels)
    # for w in terms_only:
    #     all_words.append(w)

for row in results:
    post = row[0]
    labels = row[1]
    # process = Tokenizer.tokenize(Preprocess.preprocess(post))
    # terms_only = [term for term in process if term not in stop and len(term)>2]
    #terms_tagged = pos_tag([stemmer.stem(word) for word in terms_only])
    # tests.append((terms_only, labels))
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

vectorizer = TfidfVectorizer(min_df=3,
                             max_df = 0.7,
                             sublinear_tf=True,
                             preprocessor=preprocesser,
                             tokenizer=tokenizer,
                             analyzer='word',
                             stop_words='english',
                             use_idf=True)

train_vectors = vectorizer.fit_transform(train_data)
test_vectors = vectorizer.transform(test_data)

classifier_linearSvc = LinearSVC()
t0 = time.time()
classifier_linearSvc.fit(train_vectors, train_label)
t1 = time.time()
prediction_linearSvc = classifier_linearSvc.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1

print("Results for LinearSVC")
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
print(classification_report(test_label, prediction_linearSvc))

conn.close()