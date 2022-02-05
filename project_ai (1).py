import numpy as np
import re
import pandas as pd

file=pd.read_csv("tweets",sep='\t')
file=file.sample(frac=1).reset_index(drop=True)
file.shape
file['label'].value_counts()
file.head()

f=0
for f in range (len(file)):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                           "]+", flags = re.UNICODE)
    file["tweets"][f]=regrex_pattern.sub(r'',file["tweets"][f])

from pyarabic.araby import strip_harakat
from pyarabic.araby import strip_tatweel, strip_shadda

for g in range (len(file)):
     file["tweets"][g]=strip_harakat(file["tweets"][g])
     file["tweets"][g]=strip_shadda(file["tweets"][g])
     file["tweets"][g]=strip_tatweel(file["tweets"][g])

import arabicstopwords.arabicstopwords as stp
import unicodedata
import sys
for g in range (len(file)):
     file["tweets"][g]=re.sub(r'[!"\$%-&\'()*+٪,\↺ღ؛•ೋೋ↝⁽₎».\/˓٭:;=#@؟\[\\\]⌗^_`!!⇣⇟ღ{|}~༻༺،ًُ•°˚˚°]',' ',file["tweets"][g])
     file["tweets"][g]=re.sub(r'[^\w\s]',' ', file["tweets"][g])
     file["tweets"][g]=re.sub(r'[@A-Za-z0-9_ـــــــــــــ]',' ',file["tweets"][g])

  for g in range (len(file)):
        file["tweets"][g] = re.sub(r'[\d+]+.?[\d+]+',' ', file["tweets"][g])

from nltk.corpus import stopwords
stop_words = stopwords.words('arabic')

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = stopwords.words('arabic')
for i in range(len(file)):
    text_tokens = word_tokenize(file["tweets"][i])
    file["tweets"][i] = [word for word in text_tokens if not word in stopwords.words()]
    file["tweets"][i]=' '.join( file["tweets"][i])
    print( file["tweets"][i])

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer()
X = tfidfconverter.fit_transform(file["tweets"])
feature_names = tfidfconverter.get_feature_names()

from sklearn.model_selection import KFold 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
X = X.toarray()
k = 20
kf = KFold(n_splits=k, shuffle=True)
model = LogisticRegression(solver= 'liblinear')
acc_score = []
f_score=[]
for train_index, test_index in kf.split(file):
    X_train = X[train_index, :]
    X_test = X[test_index, :]
    y_train = file.loc[train_index]['label']
    y_test = file.loc[test_index]['label']
    model.fit(X_train,y_train)
    pred_values = model.predict(X_test)   
    acc = accuracy_score(y_test, pred_values)
    acc_score.append(acc)
    f_score.append(f1_score(y_test, pred_values, pos_label='pos'))
avg_acc_score = sum(acc_score)/k
#print('accuracy of each fold - {}'.format(acc_score))
print('Avg accuracy of Logistic regression: {}'.format(avg_acc_score))
print('f1_score: {}'.format(sum(f_score)/k))


from sklearn.tree import DecisionTreeClassifier
k = 20
kf = KFold(n_splits=k, shuffle=True)
model = DecisionTreeClassifier(criterion='gini',max_depth=None,random_state=40)
acc_score = []
f_score=[]
for train_index, test_index in kf.split(file):
    X_train = X[train_index, :]
    X_test = X[test_index, :]
    y_train = file.loc[train_index]['label']
    y_test = file.loc[test_index]['label']
    model.fit(X_train,y_train)
    pred_values = model.predict(X_test)
    acc = accuracy_score(y_test, pred_values)
    acc_score.append(acc)
    f_score.append(f1_score(y_test, pred_values, pos_label='pos'))

avg_acc_score = sum(acc_score)/k
#print('accuracy of each fold - {}'.format(acc_score))
print('Avg accuracy of Decision tree: {}'.format(avg_acc_score))
print('f1_score: {}'.format(sum(f_score)/k))


from sklearn import svm
k = 20
kf = KFold(n_splits=k, shuffle=True)
model=svm.SVC(kernel ='linear', C=1.0)
acc_score = []
f_score=[]
for train_index, test_index in kf.split(file):
    X_train = X[train_index,:]
    X_test = X[test_index,:]
    y_train = file.loc[train_index]['label']
    y_test = file.loc[test_index]['label']
    model.fit(X_train,y_train)
    pred_values = model.predict(X_test)
    acc = accuracy_score(y_test, pred_values)
    acc_score.append(acc)
    f_score.append(f1_score(y_test, pred_values, pos_label='pos'))

avg_acc_score = sum(acc_score)/k
print('Avg accuracy of SVM: {}'.format(avg_acc_score))
print('f1_score: {}'.format(sum(f_score)/k))