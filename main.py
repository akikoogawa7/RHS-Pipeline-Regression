#%%
from sklearn import linear_model
from sklearn import svm
from get_data import split_data
import numpy as np

classifiers = [
    svm.SVR(),
    linear_model.SGDRegressor(),
    linear_model.BayesianRidge(),
    linear_model.LassoLars(),
    linear_model.ARDRegression(),
    linear_model.PassiveAggressiveRegressor(),
    linear_model.TheilSenRegressor(),
    linear_model.LinearRegression()]

X, y, X_train, y_train, X_test, y_test, X_val, y_val = split_data()

