import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy, scipy
from sklearn.cross_validation import cross_val_score,KFold

data = pd.read_csv('training_for_regression.csv')

feature_columns = ['TotalIndeks', 'SOV', 'SentScore', 'SOVp','SOVn']
X = data[feature_columns]
y = data.Result

linearReg = LinearRegression()
linearReg.fit(X, y)

intercept = linearReg.intercept_
coefficient = linearReg.coef_
rsquare = linearReg.score(X,y)

print("intercept: ", intercept)
print("coefficient: ", coefficient)
print("R square: ", rsquare)

import csv
def performance_eval(clf, X, y, K):
    validator = KFold (len(y), K, shuffle=True, random_state=2)
    score = cross_val_score(clf, X, y, cv=validator, scoring='mean_absolute_error')
    print(score)
    print(numpy.mean(score))
    resultFile = open("mae.csv",'w')
    wr = csv.writer(resultFile, dialect='excel-tab')
    for rows in score:
        wr.writerow([rows])

performance_eval(linearReg, X, y, 5)


# data_sent = data['SentScore']
# data_result = data['Result']
# coefficient_sent = linearReg.coef_[3]
# plt.scatter(data_sent, data_result, color='blue')
# plt.plot([0,3],[intercept,coefficient_sent*3+intercept],'r')
# plt.title('Linear Regression', fontsize = 20)
# plt.xlabel('X', fontsize = 15)
# plt.ylabel('Y', fontsize = 15)
# plt.show()

a = data.TotalIndeks
b = data.TotalTweet
c = data.SOV
d = data.SentScore
e = data.SOVp
f = data.SOVn
print("analisis korelasi:")
print(scipy.stats.pearsonr(a, y))
print(scipy.stats.pearsonr(b, y))
print(scipy.stats.pearsonr(c, y))
print(scipy.stats.pearsonr(d, y))
print(scipy.stats.pearsonr(e, y))
print(scipy.stats.pearsonr(f, y))