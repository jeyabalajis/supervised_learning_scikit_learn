# Time Series Data

## Feature Engineering

- Smoothing over time: Removes short-term noise, while retaining the general pattern.
- Instead of averaging over _all time_, we can do a _local average_. (Smoothing your timeseries)

![image](https://github.com/jeyabalajis/supervised_learning_scikit_learn/assets/15995686/2f28a286-215b-4db4-8090-df9ab6004b1a)

   
## Regression vs Correlation

Regression is similar to calculating correlation, with some key differences:
- Regression: A process that results in a formal model of the data
- Correlation: A statistic that describes the data. Less information than regression model.

## Regression Models with scikit-learn
```
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
model.predict(X)
```
> If you have noisy or correlated variables, `Ridge` regression is good to apply.

### Metrics
#### R-Squared
> R-squared = 1 - [error(model) / variance(testdata)]
```
from sklearn.metrics import r2_score
print(r2_score(y_predicted, y_test))
```

## Advanced Time Series Prediction

- Simple Interpolation
- Rolling Window (Each point represents the % change over a previous window) [Standardize the mean and variance over time] 

![image](https://github.com/jeyabalajis/supervised_learning_scikit_learn/assets/15995686/9d480225-901d-496e-ae70-ff310c0b9f84)
