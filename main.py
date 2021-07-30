#%%
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import cross_val_score
from get_data import split_data
import numpy as np

classifiers = {
    'svm': svm.SVR(),
    'sgd': linear_model.SGDRegressor(),
    'bayesian_r': linear_model.BayesianRidge(),
    'lasso_l': linear_model.LassoLars(),
    'ard_r': linear_model.ARDRegression(),
    'passive_r': linear_model.PassiveAggressiveRegressor(),
    'theilsen_r': linear_model.TheilSenRegressor(),
    'linear_r': linear_model.LinearRegression()
}

X, y, X_train, y_train, X_test, y_test, X_val, y_val = split_data()
#%%
for model in classifiers:
    list_of_scores = []
    clf = model
    clf.fit(X_train, y_train)
    clf.predict(X_train)
    scores = cross_val_score(clf, X, y, cv=5)
    print(model, 'cross validation score:', scores)
    list_of_scores[model].append(scores)
# %%
