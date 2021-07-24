#%%
from matplotlib.colors import Normalize
from sklearn import model_selection, preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, validation_curve, KFold, GridSearchCV
from sklearn. linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Set random seed
seed = 42

#%%
# LINEAR REGRESSION
# Load data
X = pd.read_csv('rhs_regression_dataset.csv')
y = X.pop('max ultimate height')
print(X.shape, y.shape)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.8, random_state=seed)

#%%
# Normalise data
scaler = preprocessing.MinMaxScaler()
X_train_minmax= scaler.fit(X_train)
X_scaled = X_train_minmax.transform(X_train)

#%%
# Fit model
linear_regression_model = LinearRegression(normalize=True)
linear_regression_model.fit(X_train, y_train)

intercept = linear_regression_model.intercept_ # scalar (bias)
print(f'intercept: {intercept}')
coefficient = linear_regression_model.coef_ # array (weights)
print(f'coefficients: {coefficient}')

# Check cross validation score
lin_reg_scores = cross_val_score(linear_regression_model, X, y, cv=5)
print(f'cross validation scores: {lin_reg_scores}')
print("%0.2f accuracy with a standard deviation of %0.2f" % (lin_reg_scores.mean(), lin_reg_scores.std()))

y_pred = linear_regression_model.predict(X)
print(f'predicted values: {y_pred}')

# Check score on test set

#%%
# Mean squared error score
def mse_score(y_pred, y):
    mse_score = round(mean_squared_error(y, y_pred, squared=True),2)
    print(f'The mean squared error is {mse_score}')
mse_score(y_pred, y)

# Plot predictions with true labels
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

#%%
# Regularise using L2
ridge_regression = Ridge(alpha=1.0, random_state=seed)
ridge_regression.fit(X, y)
new_score = ridge_regression.score(X, y)
print(f'New score after L2 norm penalty: {new_score}')
# score on validation set and benchmark it on the test set

# Hyperparameter search
# param_grid = {
#     ''
# }

# Save best
#%%
# LOGISTIC REGRESSION
import math

# Load data
X = pd.read_csv('rhs_regression_dataset.csv')
# def round_up(n):
#     decimals=0
#     multiplier = 10 ** decimals
#     return math.ceil(n * multiplier) / multiplier
X = X[['min ultimate height', 'max ultimate height', 'min ultimate spread', 'max ultimate spread', 'min time to ultimate height', 'max time to ultimate height']].astype(int)
y = X.pop('max ultimate height')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.8, random_state=seed)

# Standardise data
st_x= StandardScaler()    
x_train= st_x.fit_transform(X_train)    
x_test= st_x.transform(X_test) 

#%%
# Fit model
logistic_regression_model = LogisticRegression(random_state=seed, max_iter=1000)
logistic_regression_model.fit(X_train, y_train)

# Score model
log_reg_scores = cross_val_score(logistic_regression_model, X, y, cv=5)
print(f'cross validation scores: {log_reg_scores}')
print("%0.2f accuracy with a standard deviation of %0.2f" % (log_reg_scores.mean(), log_reg_scores.std()))

y_pred = logistic_regression_model.predict(X)
print(f'predicted values: {y_pred}')

# Visualise to evaluate
cm = confusion_matrix(y, y_pred)
plot_confusion_matrix(logistic_regression_model, X, y_pred, display_labels=)


# Report scores
# with open('metrics.txt', 'w') as outfile:
#         outfile.write("Linear Regression Metrics:\n")
#         outfile.write("Training variance explained: %2.1f%%\n" % linear_regression_test_score)
#         outfile.write("Test variance explained: %2.1f%%\n" % linear_regression_train_score)

# print(f'train score: {linear_regression_train_score}\ntest score: {linear_regression_test_score}')

