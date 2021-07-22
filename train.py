#%%
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

seed = 42

# Load in data
df = pd.read_csv('rhs_regression_dataset.csv')
y = df.pop('max time to ultimate height')

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.8, random_state=seed)

# Fit a model on the train dataset
model = LinearRegression()
model.fit(X_train, y_train)
#%%
# Report training set score
train_score = model.score(X_train, y_train) * 100
# Report test set score
test_score = model.score(X_test, y_test) * 100

# Write scores to a file

#%%
print(train_score)
df.shape
#%%
print(len(y))
# %%
