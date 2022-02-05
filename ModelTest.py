import pickle

from sklearn import metrics
import ModelTrain

def test():
    X_train, X_test, y_train, y_test=ModelTrain.getXandY()
    poly_model = pickle.load(open('finalized_model.sav', 'rb'))
    X_train_poly ,poly_features= ModelTrain.getpolynomialX(X_train)

    # predicting on training data-set
    y_train_predicted = poly_model.predict(X_train_poly)

    # predicting on test data-set
    prediction = poly_model.predict(poly_features.fit_transform(X_test))

    print('Mean Square Error train', metrics.mean_squared_error(y_train, y_train_predicted))
    print('Mean Square Error test', metrics.mean_squared_error(y_test, prediction))