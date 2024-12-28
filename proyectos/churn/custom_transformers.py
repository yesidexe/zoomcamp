import pandas as pd
from copy import deepcopy # El deep copy me copia hiperparámetros
from sklearn.base import BaseEstimator, TransformerMixin

class YesNoTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.map_dict = {
            "Yes":1,
            "No": 0,
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        X_copy = X.copy()
        
        for j in X_copy.columns:
            X_copy[j] = X_copy[j].map(self.map_dict)
        return X_copy