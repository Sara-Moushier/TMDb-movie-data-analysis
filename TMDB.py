import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



from sklearn.preprocessing import PolynomialFeatures

import ModelTest
import ModelTrain
import PreProcessing


# Call Preprocessing and save filtered data in new csv file
#PreProcessing.preprcessing()

# Train model
#ModelTrain.train()

#
ModelTest.test()
