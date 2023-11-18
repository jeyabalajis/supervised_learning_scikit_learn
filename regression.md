# Regression

- y = ax + b
- Simple Linear regression uses one feature. y = target, x = feature. a,b = parameters/coefficients of the model (slope & intercept)

- Define error function (aka loss function) and minimize error
- One example: Residual sum of squares (ordinary least squares)
- Higher dimensions = multiple regression. Must specify coefficients for each feature and  the variable b.
- Default metric is R squared: quantifies the variance in target values, explained by the features.
- Another metric: Mean Squared Error (MSE): Mean of the sum of residual squares. RMSE takes the square root of MSE

## Cross-Validation

- Model performance is dependent on the way we split up the data
- Not representative of the model's ability to generalize unseen data
- Number of folds = k. Therefore it is called as k-fold cross validation or k-fold CV

> Cross-validation is a vital approach to evaluating a model. It maximizes the amount of data that is available to the model, as the model is not only trained but also tested on all of the available data.

![image](https://github.com/jeyabalajis/supervised_learning_scikit_learn/assets/15995686/921cc3f7-4096-43c5-8a4e-897431520f36)

## Regularization

- Large co-efficients can lead to over-fitting.
- Regularization: Penalize large co-efficients
- Ridge Regression - it penalizes large positive or negative co-efficients.
- we need to choose _alpha_ (hyperparameter)
- _alpha_ = 0 (overfitting) _large alpha_ (underfitting)
- Lasso - can select important features of a dataset

![image](https://github.com/jeyabalajis/supervised_learning_scikit_learn/assets/15995686/1c89739a-00f6-4e1e-98be-4f51c96336a5)

### Example - Using Lasso for Feature Importance

```
# Import Lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regression model
lasso = Lasso(alpha=0.3)

# Fit the model to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.fit(X, y).coef_
print(lasso_coef)
plt.bar(sales_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()
```



