from joblib import dump, load

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from custom_transformers import YesNoTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from sklearn.base import clone

"""Cargo los datos y los limpio, la variable objetivo se puede procesar acá, no importa porque igual no la necesitamso en los pipelines"""

df = pd.read_csv('data/Churn.csv')

df = df.drop(columns=['customerID'])
df['TotalCharges'] = df['TotalCharges'].replace(' ', 0)
df['TotalCharges'] = df['TotalCharges'].astype(float)

df.columns = df.columns.str.lower().str.replace(' ', '_')

df['churn'] = df['churn'].map({"No": 0, "Yes": 1})

"""Train test split (para evaluar los datos)"""

X = df.drop('churn', axis=1).copy()
y = df['churn'].copy()

x_train, x_rest, y_train, y_rest = train_test_split(X, y, train_size=0.6, random_state=42, stratify=y)
x_valid, x_test, y_valid, y_test = train_test_split(x_rest, y_rest, test_size=0.5, random_state=42, stratify=y_rest)

print("Tamaño del conjunto de entrenamiento:", x_train.shape)
print("Tamaño del conjunto de validación:", x_valid.shape)
print("Tamaño del conjunto de prueba:", x_test.shape)

"""Pipelines"""

yes_no = ColumnTransformer([
    ('yes_no',
    YesNoTransformer(),
    ['partner','dependents','phoneservice','paperlessbilling'])
])

one_hot_encoding = ColumnTransformer([
    ('one_hot_encode',
    OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
    ['gender', 'multiplelines', 'internetservice', 'onlinesecurity',
        'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv',
        'streamingmovies', 'contract', 'paymentmethod'])
])

scaler = ColumnTransformer([
    ('scaler',
    MinMaxScaler(),
    ['tenure', 'monthlycharges', 'totalcharges'])
])

passthrough = ColumnTransformer([
    ('passthrough',
    'passthrough',
    ['seniorcitizen'])
])

feature_ingineering_pipe = Pipeline([
        (
            "features",
            FeatureUnion(
                [
                    ("yes_no", yes_no),
                    ("encoding", one_hot_encoding),
                    ("scaler", scaler),
                    ("pass", passthrough)
                ]
            ),
        )
])

model_pipe = Pipeline([
        ('Feature_engineering',clone(feature_ingineering_pipe)),
        ('model',LogisticRegression())
])

"""model testing"""

df_x = pd.concat([x_train,x_valid])
df_y = pd.concat([y_train, y_valid])

model_pipe.fit(df_x,df_y)

test_pred_y = model_pipe.predict(x_test)
print(f'accuracy: {accuracy_score(y_test,test_pred_y)}, recall: {recall_score(y_test,test_pred_y)}')

"""guardamos el modelo"""

dump(model_pipe, 'modelo_churn.joblib')