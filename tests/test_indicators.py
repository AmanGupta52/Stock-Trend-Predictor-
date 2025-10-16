import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from technical_indicators import TechnicalIndicators

class TestTechnicalIndicators:
    """Test cases for TechnicalIndicators"""
    
    def setup_method(self):
        """Setup before each test"""
        self.indicators = TechnicalIndicators()
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        self.sample_data = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
    
    def test_compute_all_indicators(self):
        """Test computing all indicators"""
        result = self.indicators.compute_all_indicators(self.sample_data)
        
        assert not result.empty, "Result should not be empty"
        assert self.indicators.indicators_computed, "Should mark as computed"
        
        # Check that some indicators were added
        expected_indicators = ['SMA_20', 'EMA_12', 'RSI_14', 'MACD', 'BB_Upper']
        for indicator in expected_indicators:
            assert indicator in result.columns, f"Missing indicator: {indicator}"
    
    def test_moving_averages(self):
        """Test moving average calculations"""
        result = self.indicators._compute_moving_averages(self.sample_data)
        
        # Test SMA
        assert 'SMA_20' in result.columns, "Should have SMA_20"
        assert 'SMA_50' in result.columns, "Should have SMA_50"
        
        # Test EMA
        assert 'EMA_12' in result.columns, "Should have EMA_12"
        assert 'EMA_26' in result.columns, "Should have EMA_26"
        
        # Test crossovers
        assert 'SMA_20_50_Crossover' in result.columns, "Should have crossover"
        assert 'EMA_12_26_Crossover' in result.columns, "Should have EMA crossover"
    
    def test_rsi_calculation(self):
        """Test RSI calculation"""
        result = self.indicators._compute_rsi(self.sample_data)
        
        assert 'RSI_14' in result.columns, "Should have RSI_14"
        assert 'RSI_Signal' in result.columns, "Should have RSI signal"
        
        # RSI should be between 0 and 100
        rsi_values = result['RSI_14'].dropna()
        assert (rsi_values >= 0).all() and (rsi_values <= 100).all(), "RSI should be 0-100"
    
    def test_macd_calculation(self):
        """Test MACD calculation"""
        result = self.indicators._compute_macd(self.sample_data)
        
        assert 'MACD' in result.columns, "Should have MACD"
        assert 'MACD_Signal' in result.columns, "Should have MACD signal"
        assert 'MACD_Histogram' in result.columns, "Should have MACD histogram"
        assert 'MACD_Crossover' in result.columns, "Should have MACD crossover"
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        result = self.indicators._compute_bollinger_bands(self.sample_data)
        
        assert 'BB_Upper' in result.columns, "Should have BB Upper"
        assert 'BB_Middle' in result.columns, "Should have BB Middle"
        assert 'BB_Lower' in result.columns, "Should have BB Lower"
        assert 'BB_Width' in result.columns, "Should have BB Width"
        assert 'BB_Position' in result.columns, "Should have BB Position"
        
        # BB Upper should be greater than BB Lower
        bb_upper = result['BB_Upper'].dropna()
        bb_lower = result['BB_Lower'].dropna()
        assert (bb_upper > bb_lower).all(), "BB Upper should be greater than BB Lower"
    
    def test_obv_calculation(self):
        """Test OBV calculation"""
        result = self.indicators._compute_obv(self.sample_data)
        
        assert 'OBV' in result.columns, "Should have OBV"
        assert not result['OBV'].isnull().all(), "OBV should not be all null"
    
    def test_price_features(self):
        """Test price feature calculations"""
        result = self.indicators._compute_price_features(self.sample_data)
        
        expected_features = ['Daily_Return', 'Log_Return', 'Price_Change', 
                           'High_Low_Ratio', 'Close_Open_Ratio']
        
        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"
    
    def test_volatility_features(self):
        """Test volatility feature calculations"""
        result = self.indicators._compute_volatility_features(self.sample_data)
        
        assert 'Volatility_5D' in result.columns, "Should have 5D volatility"
        assert 'Volatility_20D' in result.columns, "Should have 20D volatility"
        assert 'Volatility_Ratio' in result.columns, "Should have volatility ratio"