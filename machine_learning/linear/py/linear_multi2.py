import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import sklearn.metrics as mt 


data = pd.read_csv(r'...\machine_learning\linear\data\data_multi.csv')

deneyim_yas = data.loc[:,['deneyim','yas']].values
maas = data['maas'].values.reshape(-1,1)  # type: ignore



#algorithm
reg = lm.LinearRegression()

#data split
x_train,x_test,y_train,y_test = ms.train_test_split(deneyim_yas,maas, test_size=0.2,random_state=15)

#train
reg.fit(x_train,y_train)

#predict
y_pred = reg.predict(x_test)
print('deneyimler ve yaşlar :',x_test)
print('predict :',y_pred)

#score
score = mt.r2_score(y_test,y_pred)
print('score :',score)

#graph
plt.scatter(deneyim_yas[:,1],maas,color='r')
plt.scatter(x_test[:,1],y_pred,color='b')  # type: ignore
plt.show()