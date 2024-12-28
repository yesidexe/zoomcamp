# Importaciones estándar de Python
import logging
from typing import Tuple, Dict

# Bibliotecas externas
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import recall_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

# Guardado de modelos
from joblib import dump

# Módulos personalizados
from custom_transformers import YesNoTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChurnPredictor:    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model_pipeline = None
        self.feature_columns = {
            # Variables del one hot encoding
            'categorical': ['gender', 'multiplelines', 'internetservice',
                            'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies', 'contract', 'paymentmethod', 'partner', 'dependents', 'phoneservice', 'paperlessbilling'],
            # Varibales para escalar
            'numerical': ['tenure', 'monthlycharges', 'totalcharges'],
            # Variables para omitir
            'passthrough': ['seniorcitizen']
        }
    
    def load_and_process_data(self, filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
        try:
            df = pd.read_csv(filepath)
            logging.info(f"Data loaded successfully. {filepath}")
            
            df=df.drop(columns=['customerID'])
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            df['totalcharges'] = pd.to_numeric(df['totalcharges'].replace(' ', '0'), errors='coerce').fillna(0)

            df['churn'] = (df['churn'].str.lower() == 'yes').astype(int)
            
            X = df.drop('churn', axis=1)
            y = df['churn']
            
            return X, y
        
        except Exception as e:
            logging.error(f"Error al cargar los datos: {str(e)}")
            raise
        
    def create_pipeline(self) -> Pipeline:
        try:
            # yes_no = YesNoTransformer()            
            one_hot_encoding = OneHotEncoder(sparse_output=False, handle_unknown='ignore')            
            scaler = MinMaxScaler()
            
            feature_ingineering_pipe = ColumnTransformer(
                transformers=[
                    ('scaler', scaler, self.feature_columns['numerical']),
                    ('one_hot_encoding', one_hot_encoding, self.feature_columns['categorical']),
                    #('yes_no', yes_no, self.feature_columns['yes_no']),
                    ('passthrough', 'passthrough', self.feature_columns['passthrough'])
                ]
            )
            
            # pipeline completo
            self.model_pipeline = Pipeline([
                ('Feature_engineering', feature_ingineering_pipe),
                ('model', HistGradientBoostingClassifier(
                    random_state=self.random_state,
                    max_depth=5,
                    learning_rate=0.1,
                    max_iter=100
                ))
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