import mysql.connector
import pickle
import numpy
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from preprocesser import Preprocess
from sklearn.cross_validation import cross_val_score,KFold

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

sql = "SELECT post, label FROM `training_data` WHERE `skipstatus`=0 order by `id` asc"
c.execute(sql)
results_tests= c.fetchmany(size=400)
results_train = c.fetchall()
for row in results_train:
    post = row[0]
    labels = row[1]
    train_data.append(post)
    train_label.append(labels)

train_vectors_tf = vectorizer_tf.fit_transform(train_data)
train_vectors_tfidf = vectorizer_tfidf.fit_transform(train_data)

def classify_linear(train_vector, train_label):
    classifier_linearSvc = SVC(kernel='linear')
    linear_clf = classifier_linearSvc.fit(train_vector, train_label)
    save_clf1 = open("linearSvcTfidf.pickle","wb")
    pickle.dump(linear_clf,save_clf1)
    save_clf1.close()
    return linear_clf

def classify_rbf(train_vector, train_label):
    classifier_RBFSvc = SVC(kernel='rbf', gamma=2)
    rbf_clf = classifier_RBFSvc.fit(train_vector, train_label)
    save_clf2 = open("RBFSvcTfidf.pickle","wb")
    pickle.dump(rbf_clf,save_clf2)
    save_clf2.close()
    return rbf_clf

def classify_poly(train_vector, train_label):
    classifier_polySvc = SVC(kernel='poly', degree=2, gamma=2)
    poly_clf = classifier_polySvc.fit(train_vector, train_label)
    save_clf3 = open("polySvcTfidf.pickle","wb")
    pickle.dump(poly_clf,save_clf3)
    save_clf3.close()
    return poly_clf

def classify_linear_tf(train_vector, train_label):
    classifier_linearSvc_tf = SVC(kernel='linear')
    linear_clf_tf = classifier_linearSvc_tf.fit(train_vector, train_label)
    save_clf1 = open("linearSvcTf.pickle","wb")
    pickle.dump(linear_clf_tf,save_clf1)
    save_clf1.close()
    return linear_clf_tf

def classify_rbf_tf(train_vector, train_label):
    classifier_RBFSvc_tf = SVC(kernel='rbf', gamma=2)
    rbf_clf_tf = classifier_RBFSvc_tf.fit(train_vector, train_label)
    save_clf2 = open("RBFSvcTf.pickle","wb")
    pickle.dump(rbf_clf_tf,save_clf2)
    save_clf2.close()
    return rbf_clf_tf

def classify_poly_tf(train_vector, train_label):
    classifier_polySvc_tf = SVC(kernel='poly', degree=2, gamma=2)
    poly_clf_tf = classifier_polySvc_tf.fit(train_vector, train_label)
    save_clf3 = open("polySvcTf.pickle","wb")
    pickle.dump(poly_clf_tf,save_clf3)
    save_clf3.close()
    return poly_clf_tf

#kode dibawah berikut hanya dijalankan 1x saja
'''
result_linear_tf = classify_linear_tf(train_vectors_tf,train_label)
result_linear_tfidf = classify_linear(train_vectors_tfidf,train_label)
result_rbf_tf = classify_rbf_tf(train_vectors_tf,train_label)
result_rbf_tfidf = classify_rbf(train_vectors_tfidf,train_label)
result_poly_tf = classify_poly_tf(train_vectors_tf,train_label)
result_poly_tfidf = classify_poly(train_vectors_tfidf,train_label)

#evaluasi performa
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

train_label_array = numpy.array(train_label)
performance_eval(result_linear_tfidf, train_vectors_tf, train_label_array,10)
'''