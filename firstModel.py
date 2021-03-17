import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression


weight_height_data = pd.read_csv('/home/zera/Documents/fleventy/ml-test-practice/csvFiles/weight-height.csv')
weight_height_data.dropna(inplace= True)
data_weight = weight_height_data['Weight']
data_height = weight_height_data['Height']

x = np.array(data_weight)
x = x.reshape(-1,1)
y = np.array(data_height)
y = y.reshape(-1,1)


x_train, x_test, y_train , y_test = train_test_split(x, y , test_size = 0.044, random_state =1)



model_1 = LinearRegression()
model_1.fit(x_train,y_train)

Y_predict = model_1.predict(x_test)

confidence = model_1.score(x_test,y_test)
print(confidence)
print(Y_predict)

plt.plot(y_test)
plt.plot(Y_predict)
plt.ylabel("Height")
plt.xlabel("Number of people")
plt.show()
