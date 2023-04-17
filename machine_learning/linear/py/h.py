import pandas as pd 
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv(r'C:\Users\Yusa\Desktop\machine_learning\linear\data\h.csv')
# m = data.columns
# print(m)
l = data.dropna(axis = 0)

# print(l)
reg = DecisionTreeRegressor()
y=reg.predict()  # type: ignore
data_feature = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

x = data[data_feature] 
x.describe()
x.head()
data_model= DecisionTreeRegressor(random_state=1)
data_model.fit(x,y)

print("Making predictions for the following 5 houses:")
print(x.head())
print("The predictions are")
print(data_model.predict(x.head()))
