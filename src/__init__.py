"""
Stock Market Trend Prediction System
A comprehensive ML system for predicting stock trends using technical analysis.
"""

__version__ = "1.0.0"
__author__ = "Stock Prediction Team"
__email__ = "aman@example.com"

from .data_fetcher import StockDataFetcher
from .data_processor import DataProcessor
from .technical_indicators import TechnicalIndicators
from .feature_engineer import FeatureEngineer
from .model_trainer import ModelTrainer
from .predictor import StockPredictor
from .visualizer import StockVisualizer
from .portfolio_analyzer import PortfolioAnalyzer
from .utils import ConfigLoader, LoggerSetup

__all__ = [
    'StockDataFetcher',
    'DataProcessor',
    'TechnicalIndicators',
    'FeatureEngineer',
    'ModelTrainer',
    'StockPredictor',
    'StockVisualizer',
    'PortfolioAnalyzer',
    'ConfigLoader',
    'LoggerSetup'
]