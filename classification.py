import mysql.connector
import pickle
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from preprocesser import Preprocess
from training import *

conn = mysql.connector.connect(user='root', password='', host='127.0.0.1', database='tweets')
c = conn.cursor()

test_data = []
test_label = []

sql = "SELECT post, label FROM `table 26` WHERE `skipstatus`=0 order by `id` asc"
c.execute(sql)
results_tests= c.fetchmany(size=400)

for row in results_tests:
    post = row[0]
    labels = row[1]
    test_data.append(post)
    test_label.append(labels)

test_vectors_tf = vectorizer_tf.transform(test_data)
test_vectors_tfidf = vectorizer_tfidf.transform(test_data)

classifier_file = open("linearSvcTf.pickle", "rb")
classifier_linear_tf = pickle.load(classifier_file)
classifier_file.close()

classifier_file = open("linearSvcTfidf.pickle", "rb")
classifier_linear_tfidf = pickle.load(classifier_file)
classifier_file.close()

classifier_file = open("RBFSvcTf.pickle", "rb")
classifier_rbf_tf = pickle.load(classifier_file)
classifier_file.close()

classifier_file = open("RBFSvcTfidf.pickle", "rb")
classifier_rbf_tfidf = pickle.load(classifier_file)
classifier_file.close()

classifier_file = open("polySvcTf.pickle", "rb")
classifier_poly_tf = pickle.load(classifier_file)
classifier_file.close()

classifier_file = open("polySvcTfidf.pickle", "rb")
classifier_poly_tfidf = pickle.load(classifier_file)
classifier_file.close()

prediction_linear_tf = classifier_linear_tf.predict(test_vectors_tf)
prediction_linear_tfidf = classifier_linear_tfidf.predict(test_vectors_tfidf)
prediction_RBF_tf = classifier_rbf_tf.predict(test_vectors_tf)
prediction_RBF_tfidf = classifier_rbf_tfidf.predict(test_vectors_tfidf)
prediction_poly_tf = classifier_poly_tf.predict(test_vectors_tf)
prediction_poly_tfidf = classifier_poly_tfidf.predict(test_vectors_tfidf)

print("Results for LinearSVC with TF:")
print(classification_report(test_label, prediction_linear_tf))
print("Results for LinearSVC with TF-IDF:")
print(classification_report(test_label, prediction_linear_tfidf))
print("Results for RBF with TF:")
print(classification_report(test_label, prediction_RBF_tf))
print("Results for RBF with TF-IDF:")
print(classification_report(test_label, prediction_RBF_tfidf))
print("Results for Polynomial with TF:")
print(classification_report(test_label, prediction_poly_tf))
print("Results for Polynomial with TF-IDF:")
print(classification_report(test_label, prediction_poly_tfidf))