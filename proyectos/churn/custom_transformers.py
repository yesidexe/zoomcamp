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
        X_copy = X.copy()        
        for j in range(X.shape[1]):
            X_copy[X.columns[j]] = X.iloc[:, j].map(self.map_dict)
        return X_copy