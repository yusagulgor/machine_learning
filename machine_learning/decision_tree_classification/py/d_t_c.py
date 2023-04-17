import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix #score nin daha isabetli ve anlamlı olabilmesi için 

data = pd.read_csv(r'C:\Users\Yusa\Desktop\machine_learning\decision_tree_classification\data\d_t_c.csv')
print(data.head())
M = data[data.diagnosis == 'M']
B = data[data.diagnosis == 'B']
"""
# #Nan preposed 32 bunların silinmesi gerek temizlenmesi
# #kaldırmak için data.drop(['Unnamed : 32','id'],axis = 1 (0 olsaydı satır) ,  inplace=True)
plt.scatter(M.radius_mean,M.texture_mean,color ='r',label='M',alpha=0.3)
plt.scatter(B.radius_mean,B.texture_mean,color ='g',label='B',alpha=0.3)
plt.legend()#vermiş olduğumuz labelleri göstermesi için
plt.show()"""

#

data.diagnosis =[1 if each == 'M' else 0 for each in data.diagnosis]
x_data = data.drop(['diagnosis'], axis=1)
y= data.diagnosis.values() # type: ignore
# normalization aykırı değerlerde kurtulmak için
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

#tr ts 
x_tarin,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
#model
dt= DecisionTreeClassifier()
#train
dt.fit(x_tarin,y_train)

y_pred =  dt.predict(x_test)

cm=confusion_matrix(y_test,y_pred)
print(cm)
