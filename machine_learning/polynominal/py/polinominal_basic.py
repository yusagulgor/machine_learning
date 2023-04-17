#C:\Users\Yusa\Desktop\machine_learning\polinominal\data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures 
import sklearn.model_selection as ms
from sklearn.metrics import r2_score
import sklearn.linear_model as lm


data = pd.read_csv(r'C:\Users\Yusa\Desktop\machine_learning\polynominal\data\polynominal_basic.csv')

araba_fiyat = data['araba_fiyat'].values.reshape(-1,1)
araba_max_hiz =data['araba_max_hiz'].values.reshape(-1,1)

polynominal_reg = PolynomialFeatures(degree=4) # 2. dereceden bir denklem

x_polynominal = polynominal_reg.fit_transform(araba_fiyat,araba_max_hiz)

reg = lm.LinearRegression()

reg.fit(x_polynominal,araba_max_hiz)

y_pred = reg.predict(x_polynominal)

score = r2_score(araba_max_hiz,y_pred)
print('score :', score)

x_polynominal_pred = polynominal_reg.fit_transform( [[1500]], araba_max_hiz[0])
y_pred2 = reg.predict(x_polynominal_pred)
print(y_pred2)


plt.plot(araba_fiyat,y_pred,color='b', label='poly')
plt.legend()
plt.scatter(araba_fiyat,araba_max_hiz,color='r')
plt.xlabel('Araç fiyat :')
plt.ylabel('max hiz :')
plt.show()