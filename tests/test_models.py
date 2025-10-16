import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_trainer import ModelTrainer
from feature_engineer import FeatureEngineer

class TestModelTrainer:
    """Test cases for ModelTrainer"""
    
    def setup_method(self):
        """Setup before each test"""
        self.model_trainer = ModelTrainer()
        
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        
        self.X_train = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples)
        })
        
        self.y_train_classification = np.random.randint(0, 2, n_samples)
        self.y_train_regression = np.random.randn(n_samples)
    
    def test_initialization(self):
        """Test model trainer initialization"""
        assert self.model_trainer.model_type == 'random_forest'
        assert self.model_trainer.random_state == 42
        assert not self.model_trainer.is_trained
        assert self.model_trainer.models == {}
        assert self.model_trainer.model_performance == {}
    
    def test_train_classification_models(self):
        """Test training classification models"""
        performance = self.model_trainer.train_models(
            self.X_train, 
            self.y_train_classification,
            problem_type='classification'
        )
        
        assert self.model_trainer.is_trained, "Should be marked as trained"
        assert len(self.model_trainer.models) > 0, "Should have trained models"
        assert len(performance) > 0, "Should have performance metrics"
        
        # Check that common models are present
        expected_models = ['random_forest', 'xgboost', 'logistic_regression']
        for model in expected_models:
            assert model in self.model_trainer.models, f"Missing model: {model}"
    
    def test_train_regression_models(self):
        """Test training regression models"""
        performance = self.model_trainer.train_models(
            self.X_train, 
            self.y_train_regression,
            problem_type='regression'
        )
        
        assert self.model_trainer.is_trained, "Should be marked as trained"
        assert len(self.model_trainer.models) > 0, "Should have trained models"
        
        # Check regression-specific metrics
        for model_name, metrics in performance.items():
            assert 'rmse' in metrics, f"Missing RMSE for {model_name}"
            assert 'r2' in metrics, f"Missing R2 for {model_name}"
    
    def test_get_best_model_classification(self):
        """Test getting best model for classification"""
        self.model_trainer.train_models(
            self.X_train, 
            self.y_train_classification,
            problem_type='classification'
        )
        
        best_model = self.model_trainer._get_best_model('classification')
        assert best_model in self.model_trainer.models, "Best model should be in models"
        assert hasattr(self.model_trainer, 'best_model'), "Should have best_model attribute"
    
    def test_get_best_model_regression(self):
        """Test getting best model for regression"""
        self.model_trainer.train_models(
            self.X_train, 
            self.y_train_regression,
            problem_type='regression'
        )
        
        best_model = self.model_trainer._get_best_model('regression')
        assert best_model in self.model_trainer.models, "Best model should be in models"
    
    def test_feature_importance(self):
        """Test feature importance calculation"""
        self.model_trainer.train_models(
            self.X_train, 
            self.y_train_classification,
            problem_type='classification'
        )
        
        importance_df = self.model_trainer.get_feature_importance('random_forest')
        
        assert not importance_df.empty, "Feature importance should not be empty"
        assert 'feature' in importance_df.columns, "Should have feature column"
        assert 'importance' in importance_df.columns, "Should have importance column"
        assert len(importance_df) == len(self.X_train.columns), "Should have importance for all features"
    
    def test_cross_validation(self):
        """Test cross-validation"""
        self.model_trainer.train_models(
            self.X_train, 
            self.y_train_classification,
            problem_type='classification'
        )
        
        cv_scores = self.model_trainer.cross_validate(
            self.X_train, 
            self.y_train_classification,
            cv_splits=3,
            problem_type='classification'
        )
        
        assert len(cv_scores) > 0, "Should have CV scores"
        for model_name, scores in cv_scores.items():
            assert len(scores) == 3, f"Should have 3 CV scores for {model_name}"
    
    def test_model_saving_loading(self, tmp_path):
        """Test model saving and loading"""
        self.model_trainer.train_models(
            self.X_train, 
            self.y_train_classification,
            problem_type='classification'
        )
        
        # Test saving
        model_path = tmp_path / "test_model.joblib"
        self.model_trainer.save_model('random_forest', str(model_path))
        assert model_path.exists(), "Model file should be created"
        
        # Test loading
        self.model_trainer.load_model('random_forest_loaded', str(model_path))
        assert 'random_forest_loaded' in self.model_trainer.models, "Loaded model should be in models"