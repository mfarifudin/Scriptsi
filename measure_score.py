import sent_model as s
import mysql.connector
import pandas as pd
import math

conn = mysql.connector.connect(user='root', password='', host='127.0.0.1', database='tweets')
c = conn.cursor()

players = ['kane','aguero','eriksen','alli','vardy','payet','mahrez','ozil','lukaku','ighalo']
index_score = [280,285,147,175,173,158,189,127,124,60]
voting_result = [0.04,0.01,0,0,0.08,0.03,0.44,0.39,0,0]
sentiment_score = []
sov_score = []
sovp_score = []
sovn_score = []

for p in players:
    counter = []
    query= 'SELECT * FROM (' + ','.join(map(str, players)) + ')'
    c.execute(query)
    results = c.fetchall()
    for row in results:
        post = row[3]
        # print(post)
        sentiment_value = s.classify(post)
        counter.append(sentiment_value)
        choice_pos = counter.count("pos")
        choice_neu = counter.count("neu")
        choice_neg = counter.count("neg")
        sent_score = math.log10((choice_pos+1)/(choice_neg+1))
        sov = (len(counter)/274715)
        sovp = choice_pos/91992
        sovn = choice_neg/4919
        sentiment_score.append(sent_score)
        sov_score.append(sov)
        sovp_score.append(sovp)
        sovn_score.append(sovn)
        print("pos:", sovp)
        print("sov:", sov)
        print("neg:", sovn)
        print("sent:", sent_score)
    counter.clear()

ScoreDataSet = list(zip(players,index_score,sov_score,sentiment_score,sovp_score,sovn_score,voting_result))
df = pd.DataFrame(data=ScoreDataSet, columns=['Nama Pemain','TotalIndeks','SOV','SentScore','SOVp','SOVn','Result'])
df.to_csv('training_for_regression.csv',index=False,header=True)