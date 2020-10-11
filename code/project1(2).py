# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 17:43:01 2020

@author: 17310
"""


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

from sklearn.metrics import f1_score



col_names=['gameId','creationTime','gameDuration','seasonId','winner','firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon',
           'firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_inhibitorKills'
           ,'t2_baronKills','t2_dragonKills','t2_riftHeraldKills']
pima=pd.read_csv("C:\\Users\\17310\\Documents\\工业大数据分析及应用\\project_1\\new_data.csv",header=None,names=col_names,index_col=False)

pima.drop(['gameId'],axis = 1, inplace = True)
pima.head()
pima.drop(['creationTime'],axis = 1, inplace = True)
pima.head()


pima=pima.iloc[1:]
pima.head()
feature_cols=['gameDuration','seasonId','firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon',
           'firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_inhibitorKills'
           ,'t2_baronKills','t2_dragonKills','t2_riftHeraldKills']


#values = pima[feature_cols]
#SimpleImputer = SimpleImputer()
#imputedData = SimpleImputer.fit_transform(values)

#scaler = MinMaxScaler(feature_range=(0, 1))
#normalizedData = scaler.fit_transform(imputedData)



"""'gameId','creationTime',"""


X=pima[feature_cols]
y=pima.winner
scaler=MinMaxScaler(feature_range=(0,1))
X=scaler.fit_transform(X)





test=pd.read_csv("C:\\Users\\17310\\Documents\\工业大数据分析及应用\\project_1\\test_set.csv", header=None,names=col_names,index_col=False)
test.drop(['gameId'],axis = 1, inplace = True)
test.head()
test.drop(['creationTime'],axis = 1, inplace = True)
test.head()

test=test.iloc[1:]
test.head()
X_testset=test[feature_cols]
y_testset=test.winner
scaler=MinMaxScaler(feature_range=(0,1))
X_testset=scaler.fit_transform(X_testset)

target_names = [ 'class 1', 'class 2']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
clf=DecisionTreeClassifier(criterion="gini",max_depth=20)
start =time.perf_counter()
clf=clf.fit(X_train, y_train)
end = time.perf_counter()
print('Trainning time of decision tree: %s Seconds'%(end-start))
y_pred1=clf.predict(X_test)
y_pred1_test=clf.predict(X_testset)
print("Accuracy of decision tree in training set:",accuracy_score(y_test,y_pred1))
print("The Accuracy of decision tree:",accuracy_score(y_testset,y_pred1_test))

print("The f1_score of decision tree:",f1_score(y_testset,y_pred1_test, average='macro'))

clf2=svm.SVC()
start =time.perf_counter()
clf2=clf2.fit(X_train, y_train)
end = time.perf_counter()
print('Trainning time of svm: %s Seconds'%(end-start))
y_pred2=clf2.predict(X_test)
y_pred2_test=clf2.predict(X_testset)
print("Accuracy of svm in training set:",accuracy_score(y_test,y_pred2))
print("The Accuracy of svm:",accuracy_score(y_testset,y_pred2_test))
print("The f1_score of svm:",f1_score(y_testset,y_pred2_test, average='macro'))

clf3 = MLPClassifier(max_iter=10000)
start =time.perf_counter()
clf3=clf3.fit(X_train, y_train)
end = time.perf_counter()
print('Trainning time of multi-layer perception: %s Seconds'%(end-start))
y_pred3=clf3.predict(X_test)
y_pred3_test=clf3.predict(X_testset)
print("Accuracy of multi-layer perception in training set:",accuracy_score(y_test,y_pred3))
print("The Accuracy of multi-layer perception:",accuracy_score(y_testset,y_pred3_test))
print("The f1_score of multi-layer perception:",f1_score(y_testset,y_pred3_test, average='macro'))

# Bagged Decision Trees for Classification - necessary dependencies
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier



# Segregate the features from the labels


kfold = model_selection.KFold(n_splits=10)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, 
random_state=7)
start =time.perf_counter()
model=model.fit(X_train, y_train)
end = time.perf_counter()
y_pred4_test=model.predict(X_testset)
print('Trainning time of BaggingClassifier: %s Seconds'%(end-start))
results = model_selection.cross_val_score(model, X_testset, y_testset, cv=kfold)
print("The accuracy of bagged decision tree:",results.mean())



estimators = []
model1 = clf
estimators.append(('decision tree', model1))
model2 = clf2
estimators.append(('svm', model2))
model3 = clf3
estimators.append(('multi-layer perception', model3))
ensemble = VotingClassifier(estimators)
start =time.perf_counter()
ensemble=ensemble.fit(X_train, y_train)
end = time.perf_counter()
y_pred5_test=ensemble.predict(X_testset)
print('Trainning time of voting classifier: %s Seconds'%(end-start))
results = model_selection.cross_val_score(ensemble, X_testset, y_testset, cv=kfold)
print("The accuracy of voting classifier:",results.mean())

print("The classification_report of decision tree:\n",classification_report(y_testset,y_pred1_test,digits=4))
print("The classification_report of svm:\n",classification_report(y_testset,y_pred2_test,digits=4))
print("The classification_report of multi-layer perception:\n",classification_report(y_testset,y_pred3_test,digits=4))
print("The classification_report of BaggingClassifier\n",classification_report(y_testset,y_pred4_test,digits=4))
print("The classification_report of voting classifier:\n",classification_report(y_testset,y_pred5_test,digits=4))

"""from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import os 
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,
                special_characters=True, feature_names = 
                feature_cols,class_names=['0','1','2'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
graph.write_png('lol.png')
Image(graph.create_png())
import matplotlib.pyplot as plt # Plt is used to display pictures
import matplotlib.image as mpimg # Mpimg is used to read pictures
lol = mpimg.imread('lol.png') # Read diabetes.png in the same directory 
#as the code
# Diabetes is already an np.array and can be processed at will
plt.imshow(lol) # Show Picture
plt.axis('off') # Do not show axes
plt.show()
"""
"""plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#matplotlib画图中中文显示会有问题，需要这两行设置默认字体
 
plt.xlabel('training time')
plt.ylabel('Accuracy')
plt.xlim(xmax=65,xmin=0)
plt.ylim(ymax=2,ymin=0)"""