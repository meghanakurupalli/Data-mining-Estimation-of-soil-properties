import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


#df = pd.read_excel(r'C:\Users\dell-pc\Desktop\proj_dataset.xlsx')  # sheetname is optional
#df.to_csv(r'C:\Users\dell-pc\Desktop\csvconverted', index=False) 
#data = pd.read_csv(r'C:\Users\dell-pc\Desktop\csvconverted')


#print(data.describe())
#print("-----------------------------------------------------------------------------")
#data.dropna(axis=1, how='any')
#print(data.describe())

print("-----------------------------------------------------------------------------")
df1 = pd.read_excel(r'C:\Users\dell-pc\Desktop\train_new.xlsx')  # sheetname is optional
df1.to_csv(r'C:\Users\dell-pc\Desktop\train_data', index=False)
data1 = pd.read_csv(r'C:\Users\dell-pc\Desktop\train_data')


df2 = pd.read_excel(r'C:\Users\dell-pc\Desktop\test.xlsx')  # sheetname is optional
df2.to_csv(r'C:\Users\dell-pc\Desktop\test_data', index=False)
data2 = pd.read_csv(r'C:\Users\dell-pc\Desktop\test_data')


print(data1.head())
print(data1.describe())
print("-----------------------------------------------------------------------------")

print(data1.shape)


#cols = ['Lay_depth_to_top', 'Lay_depth_to_bottom', 'Texture'] 

#colsRes = ['TC']

y=data1.EOC

X=data1.drop('EOC', axis=1)
from sklearn.model_selection import train_test_split


#trainArr = data1.as_matrix(cols) #training array

#trainRes = data1.as_matrix(colsRes) # training results

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=100) # initialize

rf.fit(X_train, y_train)

#testArr = data2.as_matrix(cols)

results1 = rf.predict(X_test)



#data2['predictions'] = results
print("-----------------------")
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, results1, normalize=True, sample_weight=None)
print(accuracy)
print("-----------------------")

from sklearn.metrics import classification_report 
accu1=classification_report(y_test,results1)
print(accu1)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
#rf = RandomForestClassifier(n_estimators=100) # initialize

gnb.fit(X_train, y_train)

#testArr = data2.as_matrix(cols)

results2 = gnb.predict(X_test)


from sklearn import svm

clf = svm.SVC()
clf.fit(X_train, y_train) 


#rf = RandomForestClassifier(n_estimators=100) # initialize

#rf.fit(X_train, y_train)

#testArr = data2.as_matrix(cols)

results3 = clf.predict(X_test)

#data2['predictions'] = results
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report 
accu1=classification_report(y_test,results1)
print(accu1)

accu2=classification_report(y_test,results2)
print(accu2)

accu3=classification_report(y_test,results3)
print(accu3)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#conf=confusion_matrix(y_test, results)
#conf.plot()

import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(y_test, results1, normalize=True)
plt.show()

skplt.metrics.plot_confusion_matrix(y_test, results2, normalize=True)
plt.show()

skplt.metrics.plot_confusion_matrix(y_test, results3, normalize=True)
plt.show()