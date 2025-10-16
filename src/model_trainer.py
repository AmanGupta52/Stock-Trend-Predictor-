import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import logging
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_type: str = 'random_forest', random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.models = {}
        self.model_performance = {}
        self.is_trained = False
        
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_test: pd.DataFrame = None, y_test: pd.Series = None,
                    problem_type: str = 'classification') -> Dict[str, Any]:
        """Train multiple models and return performance metrics"""
        try:
            if problem_type == 'classification':
                models = {
                    'random_forest': RandomForestClassifier(
                        n_estimators=100, 
                        random_state=self.random_state,
                        max_depth=10,
                        min_samples_split=5
                    ),
                    'xgboost': xgb.XGBClassifier(
                        n_estimators=100,
                        random_state=self.random_state,
                        max_depth=6,
                        learning_rate=0.1
                    ),
                    'logistic_regression': LogisticRegression(
                        random_state=self.random_state,
                        max_iter=1000
                    ),
                    'svm': SVC(
                        random_state=self.random_state,
                        probability=True
                    )
                }
            else:
                models = {
                    'random_forest': RandomForestRegressor(
                        n_estimators=100,
                        random_state=self.random_state,
                        max_depth=10
                    ),
                    'xgboost': xgb.XGBRegressor(
                        n_estimators=100,
                        random_state=self.random_state,
                        max_depth=6,
                        learning_rate=0.1
                    )
                }
            
            performance = {}
            
            for name, model in models.items():
                logger.info(f"Training {name}...")
                model.fit(X_train, y_train)
                self.models[name] = model
                
                # Calculate performance metrics
                if X_test is not None and y_test is not None:
                    if problem_type == 'classification':
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        performance[name] = {
                            'accuracy': accuracy,
                            'model': model
                        }
                    else:
                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        performance[name] = {
                            'mse': mse,
                            'rmse': rmse,
                            'mae': mae,
                            'r2': r2,
                            'model': model
                        }
            
            self.model_performance = performance
            self.is_trained = True
            
            # Get best model
            best_model_name = self._get_best_model(problem_type)
            self.best_model = self.models[best_model_name]
            
            logger.info(f"Model training completed. Best model: {best_model_name}")
            return performance
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
    
    def _get_best_model(self, problem_type: str) -> str:
        """Get the best performing model"""
        if problem_type == 'classification':
            return max(self.model_performance.items(), 
                      key=lambda x: x[1]['accuracy'])[0]
        else:
            return min(self.model_performance.items(), 
                      key=lambda x: x[1]['rmse'])[0]
    
    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray, 
                  X_test: np.ndarray = None, y_test: np.ndarray = None,
                  sequence_length: int = 10) -> Sequential:
        """Train LSTM model for time series prediction"""
        try:
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, X_train.shape[2])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), 
                         loss='mse', 
                         metrics=['mae'])
            
            history = model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=50,
                validation_data=(X_test, y_test) if X_test is not None else None,
                verbose=0
            )
            
            self.models['lstm'] = model
            return model
            
        except Exception as e:
            logger.error(f"Error training LSTM: {str(e)}")
            raise
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      cv_splits: int = 5, problem_type: str = 'classification') -> Dict[str, List]:
        """Perform time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        cv_scores = {}
        
        for name, model in self.models.items():
            if name != 'lstm':  # LSTM requires special handling
                scores = cross_val_score(model, X, y, cv=tscv, 
                                       scoring='accuracy' if problem_type == 'classification' else 'neg_mean_squared_error')
                cv_scores[name] = scores.tolist()
        
        return cv_scores
    
    def get_feature_importance(self, model_name: str = None) -> pd.DataFrame:
        """Get feature importance from tree-based models"""
        if model_name is None:
            model_name = list(self.models.keys())[0]
        
        model = self.models.get(model_name)
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_names = getattr(model, 'feature_names_in_', 
                                  [f'feature_{i}' for i in range(len(importance))])
            
            return pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        else:
            logger.warning(f"Model {model_name} doesn't have feature importance")
            return pd.DataFrame()
    
    def save_model(self, model_name: str, filepath: str):
        """Save trained model to file"""
        if model_name in self.models:
            joblib.dump(self.models[model_name], filepath)
            logger.info(f"Model {model_name} saved to {filepath}")
        else:
            logger.error(f"Model {model_name} not found")
    
    def load_model(self, model_name: str, filepath: str):
        """Load model from file"""
        self.models[model_name] = joblib.load(filepath)
        logger.info(f"Model {model_name} loaded from {filepath}")