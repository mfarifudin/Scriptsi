import pickle
from training import vectorizer_tfidf

classifier_file = open("linearSvcTfidf.pickle", "rb")
classifier_linear_tfidf = pickle.load(classifier_file)
classifier_file.close()

def classify(query):
    temp=[]
    temp.append(query)
    new_vector = vectorizer_tfidf.transform(temp)
    return classifier_linear_tfidf.predict(new_vector)[0]

# print(classify("wayne rooney has a good instinct on scored a goal"))
