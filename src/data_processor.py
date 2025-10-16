import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Data processing and cleaning class for stock data
    """
    
    def __init__(self, fill_method: str = 'ffill', remove_outliers: bool = True):
        self.fill_method = fill_method
        self.remove_outliers = remove_outliers
        self.processed_data = None
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess stock data
        """
        try:
            df = data.copy()
            
            # Remove duplicate indices
            df = df[~df.index.duplicated(keep='first')]
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Remove outliers if requested
            if self.remove_outliers:
                df = self._remove_outliers(df)
            
            # Ensure data is sorted by date
            df = df.sort_index()
            
            # Validate data quality
            self._validate_data(df)
            
            self.processed_data = df
            logger.info(f"Data cleaning completed. Final shape: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in data cleaning: {str(e)}")
            raise
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        
        # Count missing values before processing
        missing_before = df.isnull().sum().sum()
        
        # For OHLCV data, use forward fill then backward fill
        ohlc_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        existing_ohlc = [col for col in ohlc_columns if col in df.columns]
        
        if existing_ohlc:
            # Forward fill for OHLC data
            df[existing_ohlc] = df[existing_ohlc].fillna(method='ffill')
            # Backward fill any remaining missing values
            df[existing_ohlc] = df[existing_ohlc].fillna(method='bfill')
        
        # For technical indicators, use interpolation
        indicator_cols = [col for col in df.columns if col not in existing_ohlc]
        if indicator_cols:
            df[indicator_cols] = df[indicator_cols].interpolate(method='linear')
        
        # Drop any rows that still have missing values
        df = df.dropna()
        
        missing_after = df.isnull().sum().sum()
        logger.info(f"Missing values handled: {missing_before} -> {missing_after}")
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Remove outliers from the dataset"""
        
        if method == 'iqr':
            return self._remove_outliers_iqr(df)
        elif method == 'zscore':
            return self._remove_outliers_zscore(df)
        else:
            return df
    
    def _remove_outliers_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Don't remove outliers from price columns (they might be real spikes)
            if col not in ['Open', 'High', 'Low', 'Close']:
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df
    
    def _remove_outliers_zscore(self, df: pd.DataFrame, threshold: float = 3) -> pd.DataFrame:
        """Remove outliers using Z-score method"""
        from scipy import stats
        import numpy as np
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['Open', 'High', 'Low', 'Close']:  # Skip price columns
                z_scores = np.abs(stats.zscore(df[col]))
                df = df[z_scores < threshold]
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data quality"""
        
        # Check for infinite values
        if np.isinf(df.select_dtypes(include=[np.number])).any().any():
            logger.warning("Infinite values found in data")
            return False
        
        # Check for negative prices
        price_cols = [col for col in ['Open', 'High', 'Low', 'Close'] if col in df.columns]
        if price_cols:
            if (df[price_cols] <= 0).any().any():
                logger.warning("Negative or zero prices found")
                return False
        
        # Check for negative volume
        if 'Volume' in df.columns and (df['Volume'] < 0).any():
            logger.warning("Negative volume found")
            return False
        
        # Check date index consistency
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex")
            return False
        
        logger.info("Data validation passed")
        return True
    
    def resample_data(self, df: pd.DataFrame, frequency: str = 'D') -> pd.DataFrame:
        """Resample data to different frequency"""
        try:
            if frequency == 'D':  # Daily (no resampling needed)
                return df
            
            resample_rules = {
                'W': {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'},
                'M': {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'},
            }
            
            if frequency in resample_rules:
                resampled = df.resample(frequency).agg(resample_rules[frequency])
                # For technical indicators, take the mean
                other_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
                if other_cols:
                    resampled_other = df[other_cols].resample(frequency).mean()
                    resampled = pd.concat([resampled, resampled_other], axis=1)
                
                return resampled
            else:
                logger.warning(f"Unsupported frequency: {frequency}")
                return df
                
        except Exception as e:
            logger.error(f"Error in resampling: {str(e)}")
            return df
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various return metrics"""
        df_returns = df.copy()
        
        # Simple returns
        df_returns['Simple_Return'] = df_returns['Close'].pct_change()
        
        # Log returns
        df_returns['Log_Return'] = np.log(df_returns['Close'] / df_returns['Close'].shift(1))
        
        # Cumulative returns
        df_returns['Cumulative_Return'] = (1 + df_returns['Simple_Return']).cumprod() - 1
        
        # Rolling returns
        df_returns['Rolling_Return_5D'] = df_returns['Simple_Return'].rolling(5).mean()
        df_returns['Rolling_Return_20D'] = df_returns['Simple_Return'].rolling(20).mean()
        
        return df_returns
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Get comprehensive data summary"""
        summary = {
            'start_date': df.index.min(),
            'end_date': df.index.max(),
            'total_days': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'numeric_stats': {}
        }
        
        # Numeric statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            summary['numeric_stats'][col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median()
            }
        
        return summary