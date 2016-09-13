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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cross_validation import cross_val_score,KFold,cross_val_predict
from scipy.stats import sem
from preprocesser import Preprocess
import pickle
import numpy
import math
from statistics import mode

conn = mysql.connector.connect(user='root', password='', host='127.0.0.1', database='tweets')
c = conn.cursor()
sql = "SELECT post, label FROM `table 26` WHERE `skipstatus`=0 order by `id` desc"

tokenizer = Preprocess.tokenize_and_stem
preprocesser = Preprocess.preprocessing

vectorizer_tfidf = TfidfVectorizer(min_df=5,
                             max_df = 0.7,
                             ngram_range=(1,1),
                             preprocessor=preprocesser,
                             tokenizer=tokenizer,
                             sublinear_tf=True,
                             use_idf=True)

vectorizer_tf = CountVectorizer(ngram_range=(1,1),
                                preprocessor=preprocesser,
                                tokenizer=tokenizer)

all_words = []
documents = []
tests = []
train_data = []
train_label = []
test_data = []
test_label = []
t0 = time.time()

c.execute(sql)
results= c.fetchmany(size=1)
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
'''
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

t1 = time.time()

train_vectors_tf = vectorizer_tf.fit_transform(train_data)
test_vectors_tf = vectorizer_tf.transform(test_data)
train_vectors_tfidf = vectorizer_tfidf.fit_transform(train_data)
test_vectors_tfidf = vectorizer_tfidf.transform(test_data)

# classifier_linearSvc = SVC(kernel='rbf', gamma=2)
classifier_linearSvc = SVC(kernel='linear')

tr = classifier_linearSvc.fit(train_vectors_tfidf, train_label)

prediction_linearSvc = classifier_linearSvc.predict(test_vectors_tfidf)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1

# save_clf = open("linearSvc.pickle","wb")
# pickle.dump(tr,save_clf)
# save_clf.close()

# print("Results for LinearSVC")
# print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
# print(classification_report(test_label, prediction_linearSvc))
# print(confusion_matrix(test_label, prediction_linearSvc, labels=["pos", "neu", "neg"]))

# classifier_f = open("linearSvcTf.pickle", "rb")
# classifier_pickle = pickle.load(classifier_f)
# classifier_f.close()

def classify(query):
    temp = []
    temp.append(query)
    new_vector = vectorizer_tfidf.transform(temp)
    return classifier_linearSvc.predict(new_vector)[0]

# sql2 = "SELECT post FROM  `april` LIMIT 100 "
# c.execute(sql2)
# result_for_prediction = c.fetchall()
#
# votes=[]
# for rows in prediction_linearSvc:
#     posts = rows[0]
#     prediction = classify(posts)
#     votes.append(prediction)
#
# choice_pos = votes.count("pos")
# choice_neu = votes.count("neu")
# choice_neg = votes.count("neg")
# sent_score = math.log10((choice_pos+1)/(choice_neg+1))
# print(choice_pos)
# print(choice_neu)
# print(choice_neg)
# print(sent_score)

# import csv
# resultFile = open("output.csv",'w')
# wr = csv.writer(resultFile, dialect='excel-tab')
# for row in prediction_linearSvc:
#     wr.writerow([row])

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

train_label_array = numpy.array(train_label)

def performance_eval(clf, X, y, K):
    validator = KFold (len(y), K, shuffle=True, random_state=0)
    score1 = cross_val_score(clf, X, y, cv=validator, scoring='precision_weighted')
    score2 = cross_val_score(clf, X, y, cv=validator, scoring='recall_weighted')
    score3 = cross_val_score(clf, X, y, cv=validator, scoring='f1_weighted')
    print("Result for precision: ", score1)
    print("Average precision: ", numpy.mean(score1))
    print("Result for recall: ",score2)
    print("Average recall: ",numpy.mean(score2))
    print("Result for f1 score: ",score3)
    print("Average f1 score: ",numpy.mean(score3))

performance_eval(classifier_linearSvc, train_vectors_tf, train_label_array,10)

conn.close()