#%%
from sklearn import model_selection, preprocessing
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn. linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Set random seed
seed = 42

#%%
""" DATA """
# Load in data
X = pd.read_csv('rhs_regression_dataset.csv')
y = X.pop('max time to ultimate height')
print(X.shape, y.shape)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.8, random_state=seed)

# Standardise data
scaler = preprocessing.MinMaxScaler()
X_train_minmax= scaler.fit(X_train)
X_scaled = X_train_minmax.transform(X_train)

#%%

""" LINEAR REGRESSION MODEL """
# Fit a model on the train dataset
linear_regression_model = LinearRegression(normalize=True)
linear_regression_model.fit(X_train, y_train)

intercept = linear_regression_model.intercept_ # scalar (bias)
print(f'intercept: {intercept}')
coefficient = linear_regression_model.coef_ # array (weights)
print(f'coefficient: {coefficient}')

# Report cross validation score for training set 
scores = cross_val_score(linear_regression_model, X, y, cv=5)
print(f'cross validation scores: {scores}')
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

y_pred = linear_regression_model.predict(X)
print(f'predicted values: {y_pred}')

#%%
# MSE
def mse_score(y_pred, y):
    mse_score = round(mean_squared_error(y, y_pred, squared=True),2)
    print(f'The mean squared error is {mse_score}')
mse_score(y_pred, y)

# Plot
def plot_predictions(y_pred, y):
    samples = len(y_pred)
    plt.figure()
    plt.scatter(np.arange(samples), y_pred, c='r', label='predictions')
    plt.scatter(np.arange(samples), y, c='b', label='true labels', marker='x')
    plt.legend()
    plt.xlabel('Sample numbers')
    plt.ylabel('Values')
    plt.show()

plot_predictions(y_pred, y)

def scatter(x, y_pred):
    plt.scatter(x, y_pred)
    plt.plot(X, y_pred, color='red')
#%%
# Perform L2 regression

ridge_regression = Ridge(alpha=1.0)
ridge_regression.fit(X, y)
new_score = ridge_regression.score(X, y) * 100
print(f'New score after L2 norm penalty: {new_score}')

#%%
# Write scores to a file
with open('metrics.txt', 'w') as outfile:
        outfile.write("Linear Regression Metrics:\n")
        outfile.write("Training variance explained: %2.1f%%\n" % linear_regression_test_score)
        outfile.write("Test variance explained: %2.1f%%\n" % linear_regression_train_score)

print(f'train score: {linear_regression_train_score}\ntest score: {linear_regression_test_score}')


# Regularise

# Hyperparameter search
param_grid = {
    ''
}
# Save best

