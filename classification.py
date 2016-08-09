import mysql.connector
import pickle
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from preprocesser import Preprocess


conn = mysql.connector.connect(user='root', password='', host='127.0.0.1', database='tweets')
c = conn.cursor()

tokenizer = Preprocess.tokenize_and_stem
preprocesser = Preprocess.preprocessing

vectorizer_tfidf = TfidfVectorizer(min_df=2,
                             max_df = 0.7,
                             ngram_range=(1,1),
                             preprocessor=preprocesser,
                             tokenizer=tokenizer,
                             sublinear_tf=True,
                             use_idf=True)

vectorizer_tf = CountVectorizer(ngram_range=(1,1),
                                preprocessor=preprocesser,
                                tokenizer=tokenizer)

train_data = []
train_label = []
test_data = []
test_label = []

sql = "SELECT post, label FROM bpl_train ORDER BY id ASC LIMIT 600"
c.execute(sql)
# results_test= c.fetchmany(size=100)
results_train = c.fetchall()
for row in results_train:
    post = row[0]
    labels = row[1]
    train_data.append(post)
    train_label.append(labels)

sql2 = "SELECT post, label FROM bpl_test"
c.execute(sql2)
results_test= c.fetchall()
for row in results_test:
    post = row[0]
    labels = row[1]
    test_data.append(post)
    test_label.append(labels)

train_vectors_tf = vectorizer_tf.fit_transform(train_data)
test_vectors_tf = vectorizer_tf.transform(test_data)
train_vectors_tfidf = vectorizer_tfidf.fit_transform(train_data)
test_vectors_tfidf = vectorizer_tfidf.transform(test_data)

def classify_linear(train_vector, train_label):
    classifier_linearSvc = LinearSVC()
    linear_clf = classifier_linearSvc.fit(train_vector, train_label)
    save_clf1 = open("linearSvcTfidf.pickle","wb")
    pickle.dump(linear_clf,save_clf1)
    save_clf1.close()
    return classifier_linearSvc

def classify_rbf(train_vector, train_label):
    classifier_RBFSvc = SVC(kernel='rbf')
    rbf_clf = classifier_RBFSvc.fit(train_vector, train_label)
    save_clf2 = open("RBFSvcTfidf.pickle","wb")
    pickle.dump(rbf_clf,save_clf2)
    save_clf2.close()
    return classifier_RBFSvc

def classify_poly(train_vector, train_label):
    classifier_polySvc = SVC(kernel='poly')
    poly_clf = classifier_polySvc.fit(train_vector, train_label)
    save_clf3 = open("polySvcTfidf.pickle","wb")
    pickle.dump(poly_clf,save_clf3)
    save_clf3.close()
    return classifier_polySvc

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
'''
# result_linear_tf = classify_linear(train_vectors_tf,train_label)
# result_linear_tfidf = classify_linear(train_vectors_tfidf,train_label)
# result_rbf_tf = classify_rbf(train_vectors_tf,train_label)
# result_rbf_tfidf = classify_rbf(train_vectors_tfidf,train_label)
# result_poly_tf = classify_poly(train_vectors_tf,train_label)
# result_poly_tfidf = classify_poly(train_vectors_tfidf,train_label)
'''
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