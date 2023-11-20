# Pre-Processing

- scikit-learn requires Numeric data and no missing values.

## Encoding categorical features
- Need to convert categorical features into numeric values.
- One Hot encoding (a series of binary features) `OneHotEncoder` (or) pandas `get_dummies`
- We must split our data first before imputing, to avoid _data leakage_.

## Scaling
- Many models use some form of distance to inform them (e.g.: KNN)
- We want features to be on a similar scale
- Normalizing or standardizing (scaling and centering)
- With scaling, a model can produce an R2 of 0.619 vis-a-vis a model that produces an R2 of 0.35.

![image](https://github.com/jeyabalajis/supervised_learning_scikit_learn/assets/15995686/c4e416ba-691c-43eb-b9dc-ec99f63ebdc0)



