import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

weight_height_data = pd.read_csv('/home/zera/Documents/fleventy/ml-test-practice/csvFiles/weight-height.csv')


weight_height_data = weight_height_data.dropna(axis=0)
y = weight_height_data.Weight

features = ['Height']
x = weight_height_data[features]


y_test ,y_train = train_test_split(y,test_size = 0.33,random_state =1)

x_test ,x_train = train_test_split(x,test_size = 0.33,random_state =1)

model_1 =  LinearRegression()
model_1.fit(x_train,y_train)

YPredict = model_1.predict(x_test)

model_accuracy = mean_absolute_error(y_test,YPredict)

#print(y_test.shape,y_train.shape)
#print(YPredict)


print(model_accuracy)


#print(model_accuracy/len(y))
