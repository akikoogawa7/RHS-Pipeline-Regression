#%%
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Set random seed
seed = 42

#%%
# Load in data
df = pd.read_csv('rhs_regression_dataset.csv')
y = df.pop('max time to ultimate height')

#%%
print(df)
#%%
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.8, random_state=seed)

#%%
# Standardise data
scaler = preprocessing.MinMaxScaler()
X_train_minmax= scaler.fit(X_train)
X_scaled = X_train_minmax.transform(X_train)

#%%
# Fit a model on the train dataset
model = LinearRegression()
model.fit(X_train, y_train)

# Report training set score
train_score = model.score(X_train, y_train) * 100
# Report test set score
test_score = model.score(X_test, y_test) * 100

# Write scores to a file
with open('metrics.txt', 'w') as outfile:
        outfile.write("Training variance explained: %2.1f%%\n" % train_score)
        outfile.write("Test variance explained: %2.1f%%\n" % test_score)

print(f'train score: {train_score}\ntest score: {test_score}')



# Plot 
#%%
print(train_score)
df.shape
