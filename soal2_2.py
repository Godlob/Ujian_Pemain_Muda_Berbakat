import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

data = pd.read_csv('data.csv')
df = data[['ID','Age','Overall','Potential']].copy()
# print(df2.head())
#player selection
dfselection = df[df['Age']<=25]
dfselection = dfselection[dfselection['Overall']>=80]
dfselection = dfselection[dfselection['Potential']>=80]
df = df.drop(dfselection.index)
#Creat df of target and non-target
df['class']= 'non-target'
dfselection['class']= 'target'
dfall = pd.concat([df,dfselection]).reset_index()
# data splitting
x_train,x_test,y_train,y_test = train_test_split(
    dfall[['Age','Overall','Potential']],
    dfall['class'],
    test_size=.1
)
print('Logistic Regression')
print(cross_val_score(
    LogisticRegression(solver='lbfgs'),
    x_train,
    y_train,
    cv=3
).mean())
print('SVC')
print(cross_val_score(
    SVC(gamma='auto'),
    x_train,
    y_train,
    cv=3
).mean())
print('Random Forest Classifier')
print(cross_val_score(
    RandomForestClassifier(n_estimators=100),
    x_train,
    y_train,
    cv=3
).mean())
