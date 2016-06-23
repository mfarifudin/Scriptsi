import mysql.connector
import string
import time
import nltk
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from preprocesser import Preprocess
import pickle
import json
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from statistics import mode

conn = mysql.connector.connect(user='root', password='', host='127.0.0.1', database='tweets')
c = conn.cursor()

sql = "SELECT post, label FROM bpl_train ORDER BY id ASC LIMIT 600"

stop_words = list(stopwords.words('english'))

punctuation = list(string.punctuation)
stop = stop_words + punctuation + ['rt', 'via']

tokenizer = Preprocess.tokenize_and_stem
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
    process = tokenizer(preprocesser(post))
    # terms_only = [term for term in process if term not in stop and len(term)>2]
    # terms_tagged = pos_tag([stemmer.stem(word) for word in terms_only])
    # print(process)
    documents.append((process, labels))
    train_data.append(post)
    train_label.append(labels)
    for w in process:
        all_words.append(w)

for row in results:
    post = row[0]
    labels = row[1]
    process = tokenizer(preprocesser(post))
    # terms_only = [term for term in process if term not in stop and len(term)>2]
    #terms_tagged = pos_tag([stemmer.stem(word) for word in terms_only])
    tests.append((process, labels))
    test_data.append(post)
    test_label.append(labels)

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:1000]
# print(all_words.most_common(15))

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

feature_sets = [(find_features(tweet), category) for (tweet, category) in documents]
feature_sets_test = [(find_features(tweets), category) for (tweets, category) in tests]
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
                             max_df = 0.7,
                             sublinear_tf=True,
                             preprocessor=preprocesser,
                             tokenizer=tokenizer,
                             analyzer='word',
                             use_idf=True)

train_vectors = vectorizer.fit_transform(train_data)
test_vectors = vectorizer.transform(test_data)

classifier_linearSvc = LinearSVC()
t0 = time.time()
tr = classifier_linearSvc.fit(train_vectors, train_label)
t1 = time.time()
prediction_linearSvc = classifier_linearSvc.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1

# save_clf = open("linearSvc.pickle","wb")
# pickle.dump(tr,save_clf)
# save_clf.close()

print("Results for LinearSVC")
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
print(classification_report(test_label, prediction_linearSvc))

classifier_f = open("linearSvc.pickle", "rb")
classifier_pickle = pickle.load(classifier_f)
classifier_f.close()

def classify(query):
    temp = []
    temp.append(query)
    new_vector = vectorizer.transform(temp)
    return classifier_pickle.predict(new_vector)[0]

# votes=[]
# for row in prediction_linearSvc:
#     votes.append(row)
# choice_votes = votes.count("pos")
# choice_votes2 = votes.count("neu")
# choice_votes3 = votes.count("neg")
# conf = choice_votes / len(votes)
# conf2 = choice_votes2 / len(votes)
# conf3 = choice_votes3 / len(votes)
# total = 0.5*conf+0.35*conf2+0.15*conf3
# print(total)

# import csv
# resultFile = open("output.csv",'w')
# wr = csv.writer(resultFile, dialect='excel-tab')
# for row in prediction_linearSvc:
#     wr.writerow([row])
#
# pipeline = Pipeline([('tfidf', TfidfTransformer()),
#                      ('svm', LinearSVC())])
# pipecl = SklearnClassifier(pipeline)
# pipecl.train(training_set)
# print("LinearSVC_classifier pipeline", (nltk.classify.accuracy(pipecl, testing_set))*100)
# ininyoba= "dejan is the worst player ever"
# proses_teks=find_features(tokenizer(preprocesser(ininyoba)))
# print(pipecl.classify(proses_teks))

# save_clf = open("pipeline.pickle","wb")
# pickle.dump(pipecl, save_clf)
# save_clf.close()


# class VoteClassifier(ClassifierI):
#     def __init__(self, *classifiers):
#         self._classifiers = classifiers
#
#     def classify(self, features):
#         votes = []
#         for c in self._classifiers:
#             v = c.classify(features)
#             votes.append(v)
#         return mode(votes)
#
# voted_classifier= VoteClassifier(classifier_pickle)
#
# def sentiment(text):
#     terms = find_features(text)
#     return voted_classifier.classify(terms)
conn.close()