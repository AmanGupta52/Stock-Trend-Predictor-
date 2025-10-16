import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging
# Remove the relative import and use absolute import
# from .model_trainer import ModelTrainer

logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self, model_trainer):  # Remove type hint to avoid import issue
        self.model_trainer = model_trainer
        self.predictions = {}
        self.confidence_scores = {}
    
    def predict_next_day(self, latest_data: pd.DataFrame, 
                        model_name: str = None) -> Dict[str, Any]:
        """Predict next day's price movement"""
        try:
            if not self.model_trainer.is_trained:
                raise ValueError("No trained model available. Please train the model first.")
            
            if model_name is None:
                model_name = list(self.model_trainer.models.keys())[0]
            
            model = self.model_trainer.models[model_name]
            
            # Ensure we have the latest features
            if 'Next_Day_Close' in latest_data.columns:
                latest_data = latest_data.drop('Next_Day_Close', axis=1)
            if 'Target_Binary' in latest_data.columns:
                latest_data = latest_data.drop('Target_Binary', axis=1)
            if 'Target_Multi' in latest_data.columns:
                latest_data = latest_data.drop('Target_Multi', axis=1)
            
            # Make prediction
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(latest_data)
                prediction = model.predict(latest_data)
                
                # For classification, get confidence
                confidence = np.max(prediction_proba, axis=1)[0]
                
                # Determine trend and action
                trend = "UP" if prediction[0] == 1 else "DOWN"
                action = self._get_trading_action(prediction[0], confidence)
                
                result = {
                    'prediction': prediction[0],
                    'trend': trend,
                    'action': action,
                    'confidence': confidence,
                    'model': model_name
                }
                
            else:
                # For regression models
                prediction = model.predict(latest_data)
                current_price = latest_data['Close'].iloc[0] if 'Close' in latest_data.columns else 0
                price_change = (prediction[0] - current_price) / current_price
                
                trend = "UP" if price_change > 0 else "DOWN"
                confidence = min(abs(price_change) * 10, 1.0)  # Simple confidence based on magnitude
                action = self._get_trading_action(1 if price_change > 0 else 0, confidence)
                
                result = {
                    'predicted_price': prediction[0],
                    'current_price': current_price,
                    'price_change_pct': price_change * 100,
                    'trend': trend,
                    'action': action,
                    'confidence': confidence,
                    'model': model_name
                }
            
            self.predictions[model_name] = result
            logger.info(f"Prediction completed: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise
    
    def _get_trading_action(self, prediction: int, confidence: float) -> str:
        """Determine trading action based on prediction and confidence"""
        if prediction == 1:  # Up trend
            if confidence > 0.7:
                return "STRONG BUY"
            elif confidence > 0.5:
                return "BUY"
            else:
                return "HOLD"
        else:  # Down trend
            if confidence > 0.7:
                return "STRONG SELL"
            elif confidence > 0.5:
                return "SELL"
            else:
                return "HOLD"
    
    def predict_future_days(self, data: pd.DataFrame, days: int = 5, 
                           model_name: str = None) -> pd.DataFrame:
        """Predict multiple future days"""
        predictions = []
        current_data = data.copy()
        
        for day in range(days):
            try:
                prediction = self.predict_next_day(current_data.iloc[[-1]], model_name)
                
                # Create new row for next prediction
                new_row = current_data.iloc[[-1]].copy()
                
                # Update prices for next prediction (simplified)
                if 'predicted_price' in prediction:
                    price_change = prediction['price_change_pct'] / 100
                    new_row['Close'] = new_row['Close'] * (1 + price_change)
                    new_row['Open'] = new_row['Close'] * 0.99  # Simple estimation
                    new_row['High'] = new_row['Close'] * 1.01
                    new_row['Low'] = new_row['Close'] * 0.99
                
                predictions.append({
                    'date': pd.Timestamp.now() + pd.Timedelta(days=day+1),
                    'prediction': prediction
                })
                
                # Append to data for recursive prediction (simplified)
                current_data = pd.concat([current_data, new_row])
                
            except Exception as e:
                logger.error(f"Error in multi-day prediction for day {day+1}: {str(e)}")
                break
        
        return pd.DataFrame(predictions)
    
    def get_ensemble_prediction(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get ensemble prediction from all models"""
        predictions = []
        confidences = []
        
        for model_name in self.model_trainer.models.keys():
            try:
                result = self.predict_next_day(data, model_name)
                predictions.append(1 if result['trend'] == "UP" else 0)
                confidences.append(result['confidence'])
            except Exception as e:
                logger.warning(f"Could not get prediction from {model_name}: {str(e)}")
                continue
        
        if predictions:
            # Weighted average based on confidence
            avg_prediction = np.average(predictions, weights=confidences)
            avg_confidence = np.mean(confidences)
            
            trend = "UP" if avg_prediction > 0.5 else "DOWN"
            action = self._get_trading_action(1 if trend == "UP" else 0, avg_confidence)
            
            return {
                'trend': trend,
                'action': action,
                'confidence': avg_confidence,
                'model_count': len(predictions),
                'method': 'ensemble'
            }
        else:
            raise ValueError("No successful predictions from any model")