import statistics

import pandas as pd
import numpy as np

from numpy import shape
from sklearn import metrics, preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#### The Model
# mean
def get_mean(arr):
    return np.sum(arr) / len(arr)


# variance
def get_variance(arr, mean):
    return np.sum((arr - mean) ** 2)


# covariance
def get_covariance(arr_x, mean_x, arr_y, mean_y):
    final_arr = (arr_x - mean_x) * (arr_y - mean_y)
    return np.sum(final_arr)


# Coefficients
# m = cov(x, y) / var(x)
# b = y - m*x
def get_coefficients(x, y):
    x_mean = get_mean(x)
    y_mean = get_mean(y)
    m = get_covariance(x, x_mean, y, y_mean) / get_variance(x, x_mean)
    b = y_mean - x_mean * m
    return m, b


def linear_regression(x_train, y_train, x_test, y_test):
    prediction = []
    m, b = get_coefficients(x_train, y_train)
    for x in x_test:
        y = m * x + b
        prediction.append(y)
    return prediction


def mean_squared_error(y_prediction, y_test):
    total_error = 0
    for i in range(len(y_test)):
        total_error += ((y_prediction[i] - y_test[i]) ** 2)
    total_error /= float(len(y_test))
    return total_error




dataFrame = pd.read_csv('tmdb-movies (2).csv')
# print (dataFrame)
dataFrame['budget_adj'] = dataFrame['budget_adj'].replace(0, np.NaN)
dataFrame['revenue_adj'] = dataFrame['revenue_adj'].replace(0, np.NaN)
# print (dataFrame)
mean = dataFrame.mean(axis=0, skipna=True)
dataFrame["revenue_adj"].fillna(mean.revenue_adj, inplace=True)
dataFrame["budget_adj"].fillna(mean.budget_adj, inplace=True)
dataFrame["revenue_adj"] = dataFrame["revenue_adj"] - dataFrame["budget_adj"]
dataFrame.rename(columns={"revenue_adj": 'Profit'}, inplace=True)
# print (dataFrame)
X = dataFrame['popularity']  # Popularity
Y = dataFrame['Profit'] # Profit

X = preprocessing.minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)
Y = preprocessing.minmax_scale(Y, feature_range=(0, 1), axis=0, copy=True)

#print(X)
#print(Y)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True)

#[m, b] = get_coefficients(X, Y)
# print(m)
# print(b)

y_prediction = linear_regression(X_train, y_train, X_test, y_test)
#print(y_prediction)
#print(y_test)
#print(shape(y_test))
#print(type(y_test))
#print("hello")
Mean_Square_Error = mean_squared_error(y_prediction, y_test)
print("Mean square error of implemented " ,Mean_Square_Error)


# Linear regression==============================================================
model = LinearRegression()
X_train=np.expand_dims(X_train, axis=1)
y_train=np.expand_dims(y_train, axis=1)
X_test=np.expand_dims(X_test,axis=1)
model.fit(X_train, y_train)
prediction = model.predict(X_test)

r2_score = model.score(X_test, y_test)

#print(X_test.shape, y_test.shape)

print('Mean Square Error of built-in', metrics.mean_squared_error(np.asarray(y_test), prediction))
print("Accuracy: ",r2_score * 100, '%')


#visualize LR
plt.scatter(X_train, y_train, color='blue')

plt.plot(X_train, model.predict(X_train), color='red')
plt.title('Linear Regression')
plt.xlabel('popularity')
plt.ylabel('profit')

plt.show()
r2_score = model.score(X_test, y_test)
