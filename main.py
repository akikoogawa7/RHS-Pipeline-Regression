#%%
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import cross_val_score
from get_data import split_data
from time import perf_counter
import numpy as np

models = [
    svm.SVR(),
    linear_model.SGDRegressor(),
    linear_model.BayesianRidge(),
    linear_model.LassoLars(),
    linear_model.ARDRegression(),
    linear_model.PassiveAggressiveRegressor(),
    linear_model.TheilSenRegressor(),
    linear_model.LinearRegression(),
]

X, y, X_train, y_train, X_test, y_test, X_val, y_val = split_data()

for model in models:
    time_start = perf_counter()
    clf = model
    clf.fit(X_train, y_train)
    time_finished = perf_counter()
    fit_time = time_start - time_finished
    clf.predict(X_train)
    train_scores = cross_val_score(clf, X, y, cv=5)
    test_scores = cross_val_score(clf, X, y, cv=5)

