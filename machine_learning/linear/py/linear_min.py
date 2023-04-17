#linear doğrusal regression    (maaş,ev fiyatı tahmin)
#logistic regression    karar binary boolean trump mı bydın mı 
#polinominal reg. 
#decision tree   kırılımlar üzerinden ilerler  if ele kırmızı mı evt 4 çeker mi evt
#random forest   karar ağaçalrı.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import sklearn.metrics as mt 


data = pd.read_csv(r'C:\Users\Yusa\Desktop\machine_learning\linear\data\data_min.csv')
'''
m = data.describe()
print(m)'''

deneyim = data['deneyim'].values.reshape(-1,1) # type: ignore
maas = data['maas'].values.reshape(-1,1) # type: ignore


#algorithm
reg = lm.LinearRegression()

#data split
x_train,x_test,y_train,y_test = ms.train_test_split(deneyim,maas, test_size=0.2,random_state=0)

#train
reg.fit(x_train,y_train)

#predict

y_pred = reg.predict(x_test)

print('deneyimler :',x_test)
print('predict :',y_pred)

#score
score = mt.r2_score(y_test,y_pred)
print('score :',score)

#graph
plt.scatter(deneyim,maas,color='r')
plt.scatter(x_test,y_pred,color='b')
plt.show()

def maasThEt(deneyim):
    return reg.predict(deneyim)

while True:
    deneyim_giris = input('deneyim giriniz :')
    maas_th =  maasThEt( [ [int(deneyim_giris)] ] )
    print('verilmesi gereken maaş :',maas_th)
