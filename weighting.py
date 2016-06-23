import mysql.connector
import math
from preprocesser import *
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob as tb

conn = mysql.connector.connect(user='root', password='', host='127.0.0.1', database='tweets')
c = conn.cursor()

sql = "SELECT post FROM bpl_train ORDER BY id ASC LIMIT 10"
c.execute(sql)

tokenizer = Preprocess.tokenize_and_stem
preprocesser = Preprocess.preprocessing

text0 = "Both @Dele_Alli &amp; @RomeluLukaku9 can't stop getting involved in #BPL goals. More: https://t.co/cylc5MNJk4 https://t.co/1HPgjWHKx4"
text1 = "Just half a year ago, it was said that Hazard's better than Ronaldo! He hasn't scored for #CFC ever since! #BPL https://t.co/76Fgcv4h8e"
text2 = "#BPL Team Of The Week time again, with a few player you would expect to see more often getting a gig! @Outside90"
text3 = "'stop', 'get', 'involv', 'goal'"
text4 = "'just', 'half', 'year', 'ago', 'said', 'hazard', 'better', 'ronaldo', 'score'"
text5 = "'team', 'week', 'time', 'player', 'expect', 'get', 'gig'"

text0_p = tokenizer(preprocesser(text0))
text1_p = tokenizer(preprocesser(text1))
text2_p = tokenizer(preprocesser(text2))

# train_data = []
test_data = [text0_p,text1_p,text2_p]
x_data = [text0,text1,text2]

# results = c.fetchall()
# for row in results:
#     post = row[0]
#     process = tokenizer(preprocesser(post))
#     train_data.append(process)

def term_frequency(term, train_data):
    return train_data.count(term)

def sublinear_term_frequency(term, train_data):
    count = train_data.count(term)
    if count == 0:
        return 0
    return 1 + math.log(count)

def inverse_document_frequencies(train_data):
    idf_values = {}
    all_tokens_set = set([item for sublist in train_data for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, train_data)
        idf_values[tkn] = 1 + math.log(len(train_data)/(sum(contains_token)))
    return idf_values

def tfidf(train_data):
    idf = inverse_document_frequencies(train_data)
    tfidf_documents = []
    for document in train_data:
        doc_tfidf = []
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            doc_tfidf.append(tf * idf[term])
        tfidf_documents.append(doc_tfidf)
    return tfidf_documents

tfidf_rep = tfidf(test_data)

vectorizer = TfidfVectorizer(min_df=0,
                             sublinear_tf=True,
                             preprocessor=preprocesser,
                             tokenizer=tokenizer,
                             analyzer='word',
                             use_idf=True)

sklearn_rep = vectorizer.fit_transform(x_data)
# print(sklearn_rep[0].toarray().tolist())
# print(idf_rep['goal'])
# print(tfidf_rep)


def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product/magnitude

our_tfidf_comparisons = []
for count_0, doc_0 in enumerate(tfidf_rep):
    for count_1, doc_1 in enumerate(tfidf_rep):
        our_tfidf_comparisons.append((cosine_similarity(doc_0, doc_1), count_0, count_1))

skl_tfidf_comparisons = []
for count_0, doc_0 in enumerate(sklearn_rep.toarray()):
    for count_1, doc_1 in enumerate(sklearn_rep.toarray()):
        skl_tfidf_comparisons.append((cosine_similarity(doc_0, doc_1), count_0, count_1))

# for x in zip(sorted(our_tfidf_comparisons, reverse = True), sorted(skl_tfidf_comparisons, reverse = True)):
#     print(x)

# from textblob import TextBlob as tb
#
# def tf(word, blob):
#     return blob.words.count(word) / len(blob.words)
#
# def n_containing(word, bloblist):
#     return sum(1 for blob in bloblist if word in blob.words)
#
# def idf(word, bloblist):
#     return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))
#
# def tfidf(word, blob, bloblist):
#     return tf(word, blob) * idf(word, bloblist)
#
# document1 = tb(text3)
# document2 = tb(text4)
# document3 = tb(text5)
#
# bloblist = [document1, document2, document3]
# for i, blob in enumerate(bloblist):
#     print("Top words in document {}".format(i + 1))
#     scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
#     sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
#     for word, score in sorted_words[:11]:
#         print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))