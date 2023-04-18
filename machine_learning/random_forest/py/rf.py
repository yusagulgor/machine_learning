import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor # sınıflandırma prob çözmediğim için regres

data = pd.read_csv(r'...\machine_learning\random_forest\data\rf.csv')

x=data.iloc[:,0].values.reshape(-1,1) # type: ignore
y=data.iloc[:,1].values.reshape(-1,1) # type: ignore

rf = RandomForestRegressor(n_estimators=100,random_state=40) # 10 kaç ağaç daha fazla basamak  

rf.fit(x,y)

x2 =np.arange(min(x),max(x),0.01).reshape(-1,1)
y_perd= rf.predict(x2)

plt.scatter(x,y,color='r')
plt.plot(x2,y_perd,color= 'b')
plt.show()
