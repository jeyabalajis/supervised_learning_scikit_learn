# Pre-Processing

- scikit-learn requires Numeric data and no missing values.
- Need to convert categorical features into numeric values.
- One Hot encoding (a series of binary features) `OneHotEncoder` (or) pandas `get_dummies`
- We must split our data first before imputing, to avoid _data leakage_.  
