from joblib import dump, load
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from custom_transformers import YesNoTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import clone
import logging
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChurnPredictor:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model_pipeline = None
        self.feature_columns = {
            'yes_no': ['partner', 'dependents', 'phoneservice', 'paperlessbilling'],
            # Variables del one hot encoding
            'categorical': ['gender', 'multiplelines', 'internetservice',
                            'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies', 'contract', 'paymentmethod'],
            # Varibales para escalar
            'numerical': ['tenure', 'monthlycharges', 'totalcharges'],
            # Variables para omitir
            'passthrough': ['seniorcitizen']
        }
    
    def load_and_process_data(self, filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
        try:
            df = pd.read_csv(filepath)
            logging.info(f"Data loaded successfully. {filepath}")
            
            # Limpieza de datos
            df=df.drop(columns=['customerID'])
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', '0'), errors='coerce').fillna(0)

            # Verificar valores nulos
            null_counts = df.isnull().sum()
            if null_counts.any():
                logging.warning(f"Valores nulos encontrados:\n{null_counts[null_counts > 0]}")
            
            # Normalización de nombres de columnas
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            # Convertir variable objetivo
            df['churn'] = df['churn'].map({"No": 0, "Yes": 1})
            
            X = df.drop('churn', axis=1)
            y = df['churn']
            
            return X, y
        
        except Exception as e:
            logging.error(f"Error al cargar los datos: {str(e)}")
            raise
        
    def create_pipeline(self) -> Pipeline:
        try:
            # Transformadores
            yes_no = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='no')),
                ('yes_no',YesNoTransformer())
            ])
            
            one_hot_encoding = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('one_hot_encode', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ])
            
            scaler = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler',MinMaxScaler())
            ])
            
            passthrough = ColumnTransformer([
                ('passthrough',
                'passthrough',
                self.feature_columns['passthrough'])
            ])
            
            # Pipeline de features
            feature_ingineering_pipe = ColumnTransformer(
                transformers=[
                    ('scaler', scaler, self.feature_columns['numerical']),
                    ('one_hot_encoding', one_hot_encoding, self.feature_columns['categorical']),
                    ('yes_no', yes_no, self.feature_columns['yes_no']),
                    ('passthrough', passthrough, self.feature_columns['passthrough'])
                ]
            )
            
            # pipeline completo
            self.model_pipeline = Pipeline([
                ('Feature_engineering', clone(feature_ingineering_pipe)),
                ('model', LogisticRegression(random_state=self.random_state))
            ])
            
            return self.model_pipeline
        
        except Exception as e:
            logging.error(f"Error al crear el pipeline: {str(e)}")
            raise
    
    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, valid_size: float = 0.2) -> Dict:
        try:
            # Train-test split
            x_train, x_rest, y_train, y_rest = train_test_split(X, y, test_size=test_size + valid_size, random_state=self.random_state, stratify=y)         
            x_valid, x_test, y_valid, y_test = train_test_split(x_rest, y_rest, test_size=0.5, random_state=self.random_state, stratify=y_rest)
            
            logging.info(f"Train shape: {x_train.shape}, Valid shape: {x_valid.shape}, Test shape: {x_test.shape}")
            
            # Entrenamiento
            df_x = pd.concat([x_train, x_valid])
            df_y = pd.concat([y_train, y_valid])
            
            self.model_pipeline.fit(df_x, df_y)
            
            # Evaluación
            test_pred_y = self.model_pipeline.predict(x_test)
            
            # cross-validation
            cv_scores = cross_val_score(self.model_pipeline, df_x, df_y, cv=5)
            
            metrics = {
                'accuracy': accuracy_score(y_test, test_pred_y),
                'recall': recall_score(y_test, test_pred_y),
                # 'classification_report': classification_report(y_test, test_pred_y),
                'cv_scores_mean': cv_scores.mean(),
                'cv_scores_std': cv_scores.std()
            }
            
            logging.info(f"Métricas de evaluación:\n{metrics}")
            return metrics
        
        except Exception as e:
            logging.error(f"Error en entrenamiento y evaluación: {str(e)}")
            raise

    def save_model(self, filepath: str) -> None:
        try:
            dump(self.model_pipeline, filepath)
            logging.info(f"Modelo guardado en {filepath}")
        except Exception as e:
            logging.error(f"Error al guardar el modelo: {str(e)}")
            raise
    
def main():
    try:
        predictor = ChurnPredictor(random_state=42)
        X, y = predictor.load_and_process_data('data/Churn.csv')
        predictor.create_pipeline()
        metrics = predictor.train_and_evaluate(X, y)
        predictor.save_model('modelo_churn.joblib')
    except Exception as e:
        logging.error(f"Error en el flujo principal: {str(e)}")
        raise

if __name__ == "__main__":
    main()