# -*- coding: utf-8 -*-

import numpy as np
import sklearn
import os
import pandas as pd
# Import of support vector machine (svm)
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix



p=pd.read_csv('poker.data',sep = ',')


t=pd.read_csv('poker-hand-testing.data',sep = ',')

x1=t.iloc[1:1000,0:10]

y1=t.iloc[1:1000,10]

x=p.iloc[1:1000,0:10]
y=p.iloc[1:1000,10]


clf = svm.SVC(gamma='scale',kernel='rbf')
clf.fit(x, y)

a=clf.predict(x1)
print(clf.score(x1,y1))
print(accuracy_score(a,y1))

print(confusion_matrix(y1, a))
print(classification_report(y1, a))
