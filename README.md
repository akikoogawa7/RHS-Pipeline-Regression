# RHS-Pipeline-Regression
### Introduction
This ongoing machine learning regression project is a dummy project to apply my knowledge on AI, by comparing models and conducting hyperparameter searches that are relevant to the problem I am solving, which is to create the optimum model to estimate the maximum time taken for a plant to grow to its maximum height based on continuous variables.
<br>
This is based on a [data pipeline](https://github.com/akikoogawa7/RHS-Data-Pipeline) which I have collected and cleaned prior to this.

### Target
The aim is to measure which machine learning models perform best on unseen data. 
<br>
Target variable: max time to ultimate height.
<br>
Features: `'Full Sun', 'Sheltered', 'Generally pest free'`
<br>
This model helps to predict the time it takes for a plant to grow to its maximum height based on the above predictors.
<br>

### Models
- Linear regression (baseline)
- Decision tree regressor
- K nearest neighbors regressor
- Ridge regression

### Results
| Model                        | Hyperparameter  | Training Score | Validation Score | R2 Score |
|------------------------------|-----------------|----------------|------------------|----------|
| Linear Regression (baseline) | -               | 0.021          | -0.029           | -0.018   |
| Decision Tree Regressor      | criterion='mse' | 0.028          | -0.021           | -0.010   |
| KNN                          | n_neighbors=4   | -0.20          | -0.184           | -0.188   |
| Ridge Regression             | alpha=0.1       | 0.021          | -0.029           | -0.018   |
