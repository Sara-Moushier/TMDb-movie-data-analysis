
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

def getXandY():
    # load preprcessed data
    dataFrame = pd.read_csv("preprcessedData.csv")

    # Model
    first_column = dataFrame.pop('Profit')
    dataFrame.insert(0, 'Profit', first_column)

    X = dataFrame.iloc[:, 1:26]  # Features
    Y = dataFrame['Profit']  # Label

    # standarization
    X = (X - X.mean()) / X.std()
    Y = (Y - Y.mean()) / Y.std()
    Y = np.expand_dims(Y, axis=1)

    '''
    # normalization
    X = preprocessing.minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)
    Y = preprocessing.minmax_scale(Y, feature_range=(0, 1), axis=0, copy=True)
    print("df after normalization\n", X,Y)
    '''
    # Split the data to training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    return  X_train, X_test, y_train, y_test


def getpolynomialX(X_train):
    # Polynomial 3rd degree
    poly_features = PolynomialFeatures(degree=3)

    # Transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(X_train)
    return X_train_poly,poly_features



def train():
    X_train, X_test, y_train, y_test = getXandY()
    X_train_poly,x=getpolynomialX(X_train)

    # fit the transformed features to Linear Regression
    poly_model = linear_model.LinearRegression()
    poly_model.fit(X_train_poly, y_train)

    filename = 'finalized_model.sav'
    pickle.dump(poly_model, open(filename, 'wb'))