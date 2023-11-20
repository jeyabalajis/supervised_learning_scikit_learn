# Classification

1. Build a model
2. The model learns from the labeled data we pass to it
3. Pass unlabeled data to the model as input
4. The model predicts the label of the unseen data

## K-Nearest Neighbors

> Predict the label of a data point by looking at the k closest labeled points and taking a majority vote.
> Too large a k, simpler model (underfitting)
> Too small a k, more complex model (overfitting)

### Predicting Model Accuracy
Here is a sample code for splitting training data, fitting the model (with training data) and validating the accuracy (with test data)
```
# Import the module
from sklearn.model_selection import train_test_split

X = churn_df.drop("churn", axis=1).values
y = churn_df["churn"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))
```
### How good is your model?
- Precision: True positives / (TP+FP).
      - Not many legitimate transactions are predicted to be fraudulent (i.e. positive)
- Recall (Sensitivity): True Positives / (TP+FN). High recall: Lower false negative rate.
      - Higher recall: Most of the fraudulent transactions are classified correctly.
- Harmonic mean of Precision and Recall.

### Overfitting and Underfitting

![image](https://github.com/jeyabalajis/supervised_learning_scikit_learn/assets/15995686/323addc3-6526-4de7-8993-30fa5e18331b)

### ROC Curve

- The model predicts 1 for all the data (or) zero for all the data (This is the straight line in the ROC curve)
- If the threshold is varied, we get different true positive and false positive rates
- A line plot plots the trend.
- If the ROC Curve is above the dotted line, the model performs better than randomly guessing the class of each observation.

### Hyperparameter Tuning

- Using the optimal hyperparameters does not guarantee a high performing model.
