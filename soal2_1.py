import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
df = data[['Age','Overall','Potential']].copy()
# print(df2.head())
#player selection
dfselection = df[df['Age']<=25]
dfselection = dfselection[dfselection['Overall']>=80]
dfselection = dfselection[dfselection['Potential']>=80]
# print(dfselection)
plt.figure('figure 1',figsize=(10,8))
plt.subplot(121)
plt.scatter(df['Age'],df['Overall'],color='r',s=12,label='Non-Target')
plt.scatter(dfselection['Age'],dfselection['Overall'],color='g',s=12,label='Target')
plt.title('Age vs Overall')
plt.grid()
plt.xlabel('Age')
plt.ylabel('Overall')
plt.legend()
plt.subplot(122)
plt.scatter(df['Age'],df['Potential'],color='r',s=12,label='Non-Target')
plt.scatter(dfselection['Age'],dfselection['Potential'],color='g',s=12,label='Target')
plt.title('Age vs Potential')
plt.grid()
plt.xlabel('Age')
plt.ylabel('Potential')
plt.legend()
plt.show()
