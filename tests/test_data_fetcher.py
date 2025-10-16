import pytest
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_fetcher import StockDataFetcher

class TestStockDataFetcher:
    """Test cases for StockDataFetcher"""
    
    def setup_method(self):
        """Setup before each test"""
        self.fetcher = StockDataFetcher(cache_duration=0)  # Disable cache for tests
    
    def test_get_stock_data_basic(self):
        """Test basic stock data fetching"""
        data = self.fetcher.get_stock_data("RELIANCE.NS", period="1mo")
        
        assert not data.empty, "Data should not be empty"
        assert isinstance(data, pd.DataFrame), "Should return DataFrame"
        assert 'Close' in data.columns, "Should have Close column"
        assert 'Open' in data.columns, "Should have Open column"
        assert 'High' in data.columns, "Should have High column"
        assert 'Low' in data.columns, "Should have Low column"
        assert 'Volume' in data.columns, "Should have Volume column"
    
    def test_get_stock_data_invalid_ticker(self):
        """Test with invalid ticker"""
        with pytest.raises(ValueError):
            self.fetcher.get_stock_data("INVALID_TICKER", period="1mo")
    
    def test_get_multiple_stocks(self):
        """Test fetching multiple stocks"""
        tickers = ["RELIANCE.NS", "TCS.NS"]
        data_dict = self.fetcher.get_multiple_stocks(tickers, period="1mo")
        
        assert isinstance(data_dict, dict), "Should return dictionary"
        assert len(data_dict) == len(tickers), "Should return data for all tickers"
        
        for ticker, data in data_dict.items():
            assert not data.empty, f"Data for {ticker} should not be empty"
            assert 'Close' in data.columns, f"{ticker} should have Close column"
    
    def test_get_company_info(self):
        """Test company info fetching"""
        info = self.fetcher.get_company_info("RELIANCE.NS")
        
        assert isinstance(info, dict), "Should return dictionary"
        assert 'name' in info, "Should have company name"
        assert 'sector' in info, "Should have sector information"
    
    def test_data_columns(self):
        """Test data contains expected columns"""
        data = self.fetcher.get_stock_data("RELIANCE.NS", period="1mo")
        
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_columns:
            assert col in data.columns, f"Missing column: {col}"
    
    def test_data_index(self):
        """Test data index is DatetimeIndex"""
        data = self.fetcher.get_stock_data("RELIANCE.NS", period="1mo")
        
        assert isinstance(data.index, pd.DatetimeIndex), "Index should be DatetimeIndex"
        assert data.index.is_monotonic_increasing, "Index should be sorted"