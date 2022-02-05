import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataFrame = pd.read_csv("preprcessedData.csv")
first_column = dataFrame.pop('Profit')

dataFrame.insert(0, 'Profit', first_column)


X=dataFrame.iloc[:,1:26] #Features
Y=dataFrame['Profit'] #Label

#X=preprocessing.minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)
#Y=preprocessing.minmax_scale(Y, feature_range=(0, 1), axis=0, copy=True)
print(dataFrame.shape)
print(Y)
y=Y
Y=np.expand_dims(Y, axis=1)
#Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state=42)

model = linear_model.LinearRegression()
#train
model.fit(X_train,y_train)
#test
prediction= model.predict(X_test)

r2_score = model.score(X_test,y_test)
print('Co-efficient of linear regression',model.coef_)
print('Intercept of, linear regression model',model.intercept_)
#print(X_test.shape, y_test.shape)
print('Mean Square Error train', metrics.mean_squared_error(y_train, y_train))
print('Mean Square Error test', metrics.mean_squared_error(y_test, prediction))
print(r2_score*100,'%')



