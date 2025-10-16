import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from predictor import StockPredictor
from model_trainer import ModelTrainer

class TestStockPredictor:
    """Test cases for StockPredictor"""
    
    def setup_method(self):
        """Setup before each test"""
        # Create and train a simple model
        self.model_trainer = ModelTrainer()
        
        np.random.seed(42)
        n_samples = 100
        
        X_train = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'Close': np.random.uniform(100, 200, n_samples)
        })
        
        y_train = np.random.randint(0, 2, n_samples)
        
        self.model_trainer.train_models(
            X_train, y_train,
            problem_type='classification'
        )
        
        self.predictor = StockPredictor(self.model_trainer)
        
        # Create sample prediction data
        self.sample_data = pd.DataFrame({
            'feature1': [0.5],
            'feature2': [-0.3],
            'Close': [150.0]
        })
    
    def test_initialization(self):
        """Test predictor initialization"""
        assert self.predictor.model_trainer == self.model_trainer
        assert self.predictor.predictions == {}
        assert self.predictor.confidence_scores == {}
    
    def test_predict_next_day(self):
        """Test next day prediction"""
        prediction = self.predictor.predict_next_day(self.sample_data)
        
        assert isinstance(prediction, dict), "Should return dictionary"
        assert 'trend' in prediction, "Should have trend prediction"
        assert 'action' in prediction, "Should have action recommendation"
        assert 'confidence' in prediction, "Should have confidence score"
        assert 'model' in prediction, "Should have model information"
        
        # Check trend values
        assert prediction['trend'] in ['UP', 'DOWN'], "Trend should be UP or DOWN"
        
        # Check action values
        valid_actions = ['STRONG BUY', 'BUY', 'HOLD', 'SELL', 'STRONG SELL']
        assert prediction['action'] in valid_actions, f"Invalid action: {prediction['action']}"
        
        # Check confidence range
        assert 0 <= prediction['confidence'] <= 1, "Confidence should be between 0 and 1"
    
    def test_predict_next_day_with_model_name(self):
        """Test prediction with specific model"""
        prediction = self.predictor.predict_next_day(
            self.sample_data, 
            model_name='random_forest'
        )
        
        assert prediction['model'] == 'random_forest', "Should use specified model"
    
    def test_trading_action_logic(self):
        """Test trading action logic"""
        # Test BUY signals
        assert self.predictor._get_trading_action(1, 0.8) == "STRONG BUY"
        assert self.predictor._get_trading_action(1, 0.6) == "BUY"
        assert self.predictor._get_trading_action(1, 0.4) == "HOLD"
        
        # Test SELL signals
        assert self.predictor._get_trading_action(0, 0.8) == "STRONG SELL"
        assert self.predictor._get_trading_action(0, 0.6) == "SELL"
        assert self.predictor._get_trading_action(0, 0.4) == "HOLD"
    
    def test_ensemble_prediction(self):
        """Test ensemble prediction"""
        # Create sample data for multiple models
        sample_data_multi = pd.DataFrame({
            'feature1': [0.1, 0.2, 0.3],
            'feature2': [-0.1, -0.2, -0.3],
            'Close': [150.0, 151.0, 152.0]
        })
        
        ensemble_pred = self.predictor.get_ensemble_prediction(sample_data_multi)
        
        assert isinstance(ensemble_pred, dict), "Should return dictionary"
        assert 'trend' in ensemble_pred, "Should have trend prediction"
        assert 'action' in ensemble_pred, "Should have action recommendation"
        assert 'confidence' in ensemble_pred, "Should have confidence score"
        assert 'model_count' in ensemble_pred, "Should have model count"
        assert 'method' in ensemble_pred, "Should have method information"
    
    def test_prediction_storage(self):
        """Test that predictions are stored"""
        initial_count = len(self.predictor.predictions)
        
        prediction = self.predictor.predict_next_day(self.sample_data)
        
        assert len(self.predictor.predictions) == initial_count + 1, "Should store prediction"
        assert self.sample_data.index[0] in self.predictor.predictions, "Should store by index"
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test with untrained model
        untrained_trainer = ModelTrainer()
        untrained_predictor = StockPredictor(untrained_trainer)
        
        with pytest.raises(ValueError):
            untrained_predictor.predict_next_day(self.sample_data)
        
        # Test with empty data
        empty_data = pd.DataFrame()
        with pytest.raises(Exception):
            self.predictor.predict_next_day(empty_data)