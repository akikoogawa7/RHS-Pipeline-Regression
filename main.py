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

# Create txt file of model performance
def create_report(model):
    with open(f'{model}_metrics.txt', 'w') as outfile:
        outfile.write(f"Regression Metrics:\nFit time: {fit_time}\n{clf} training scores: {train_scores}\n{clf} testing scores: {test_scores}\n\n")

for model in models:
    time_start = perf_counter()
    clf = model
    clf.fit(X_train, y_train)
    time_finished = perf_counter()
    fit_time = time_start - time_finished
    clf.predict(X_train)
    train_scores = cross_val_score(clf, X, y, cv=5)
    test_scores = cross_val_score(clf, X, y, cv=5)
    print(f'{clf}\nTrain scores:\n{train_scores}\nTest scores: {test_scores}\nFit time: {fit_time}\n\n')
    create_report(model)