#%%
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# %%
def split_data():
    seed = 42
    X = pd.read_csv('rhs_regression_dataset.csv')
    y = X.pop('max ultimate height')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.8, random_state=seed)
    return X, y, X_train, y_train, X_test, y_test, X_val, y_val

