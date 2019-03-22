import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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
#data splitting
x_train,x_test,y_train,y_test = train_test_split(
    dfall[['Age','Overall','Potential']],
    dfall['class'],
    test_size=.1
)
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
pemainbaru = pd.DataFrame({
    'name':['Andik Vermansyah','Awan Setho Raharjo','Bambang Pamungkas','Cristian Gonzales',
    'Egy Maulana Vikri', 'Evan Dimas','Febri Hariyadi','Hansamu Yama Pranata',
    'Septian David Maulana','Stefano Lilipaly'],
    'Age' :[27,22,38,43,18,24,23,24,22,29],
    'Overall': [87,75,85,90,88,85,77,82,83,88],
    'Potential':[90,83,75,85,90,87,80,85,80,86]
}
)
prediction = model.predict(pemainbaru[['Age','Overall','Potential']])
dfprediksi = pemainbaru.copy()
dfprediksi['class']=prediction
print(dfprediksi)