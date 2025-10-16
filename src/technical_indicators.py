import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging
import warnings

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    def __init__(self):
        self.indicators_computed = False
        self.original_data_shape = None
        self.timezone_handled = False
    
    def compute_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicators with robust error handling and timezone support"""
        try:
            logger.info(f"Starting indicator computation. Input shape: {df.shape}")
            self.original_data_shape = df.shape
            
            # **FIX 1: Handle timezone-aware timestamps**
            data = self._handle_timezone(df.copy())
            
            # Validate required columns
            required_cols = ['Close']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return data
            
            optional_cols = ['Open', 'High', 'Low', 'Volume']
            available_cols = [col for col in optional_cols if col in data.columns]
            logger.info(f"Available columns: {available_cols}")
            
            # Compute indicators step by step with error handling
            indicators_applied = []
            
            # Core price-based indicators (always try)
            try:
                data = self._compute_moving_averages(data)
                indicators_applied.append("Moving Averages")
            except Exception as e:
                logger.warning(f"Moving averages failed: {e}")
            
            try:
                data = self._compute_rsi(data)
                indicators_applied.append("RSI")
            except Exception as e:
                logger.warning(f"RSI computation failed: {e}")
            
            try:
                data = self._compute_macd(data)
                indicators_applied.append("MACD")
            except Exception as e:
                logger.warning(f"MACD computation failed: {e}")
            
            try:
                data = self._compute_bollinger_bands(data)
                indicators_applied.append("Bollinger Bands")
            except Exception as e:
                logger.warning(f"Bollinger Bands failed: {e}")
            
            # Volume-based indicators (if Volume available)
            if 'Volume' in data.columns:
                try:
                    data = self._compute_obv(data)
                    indicators_applied.append("OBV")
                except Exception as e:
                    logger.warning(f"OBV computation failed: {e}")
                
                try:
                    data = self._compute_volume_indicators(data)
                    indicators_applied.append("Volume Indicators")
                except Exception as e:
                    logger.warning(f"Volume indicators failed: {e}")
            
            # Volatility indicators (if High/Low available)
            if all(col in data.columns for col in ['High', 'Low']):
                try:
                    data = self._compute_atr(data)
                    indicators_applied.append("ATR")
                except Exception as e:
                    logger.warning(f"ATR computation failed: {e}")
                
                try:
                    data = self._compute_stochastic(data)
                    indicators_applied.append("Stochastic")
                except Exception as e:
                    logger.warning(f"Stochastic computation failed: {e}")
            
            # Additional features
            try:
                data = self._compute_price_features(data)
                indicators_applied.append("Price Features")
            except Exception as e:
                logger.warning(f"Price features failed: {e}")
            
            try:
                data = self._compute_volatility_features(data)
                indicators_applied.append("Volatility Features")
            except Exception as e:
                logger.warning(f"Volatility features failed: {e}")
            
            # Handle any NaN values introduced
            data = self._handle_indicator_nans(data)
            
            # Validate output
            final_shape = data.shape
            logger.info(f"Indicators computed: {', '.join(indicators_applied)}")
            logger.info(f"Final shape: {final_shape} (original: {self.original_data_shape})")
            
            if len(indicators_applied) > 0:
                self.indicators_computed = True
                logger.info("✅ Technical indicators computed successfully")
            else:
                logger.warning("⚠️ No indicators could be computed")
                self.indicators_computed = False
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Critical error in indicator computation: {str(e)}")
            self.indicators_computed = False
            return df.copy()  # Return original data unchanged
    
    def _handle_timezone(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle timezone-aware timestamps safely"""
        if data.index.tz is not None and not self.timezone_handled:
            logger.info("Converting timezone-aware index to naive timestamps...")
            try:
                # Store original for reference
                original_index = data.index
                # Remove timezone information
                data.index = data.index.tz_localize(None)
                self.timezone_handled = True
                logger.info("Timezone conversion successful")
            except Exception as e:
                logger.warning(f"Timezone conversion failed: {e}")
                # Fallback: convert to UTC then remove timezone
                data.index = data.index.tz_convert('UTC').tz_localize(None)
        
        return data
    
    def _handle_indicator_nans(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle NaN values introduced by indicators"""
        # Forward fill to preserve time series continuity
        data = data.ffill()
        
        # Backward fill any remaining gaps
        data = data.bfill()
        
        # Fill remaining NaN with reasonable defaults
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['Close', 'Open', 'High', 'Low', 'Volume']:
                continue  # Don't fill original price data
            if data[col].isna().all():
                # If entire column is NaN, drop it
                data = data.drop(columns=[col])
                logger.warning(f"Dropped empty indicator column: {col}")
            else:
                # Fill with column mean or 0 for signals
                if any(signal in col for signal in ['Signal', 'Crossover']):
                    data[col] = data[col].fillna(0)
                else:
                    data[col] = data[col].fillna(data[col].mean())
        
        # Ensure no negative prices or volumes
        if 'Close' in data.columns:
            data['Close'] = data['Close'].clip(lower=0)
        if 'Volume' in data.columns:
            data['Volume'] = data['Volume'].clip(lower=0)
        
        return data
    
    def _compute_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute moving averages with validation"""
        periods = [5, 10, 20, 50]
        # Only compute long-term MA if enough data
        if len(data) >= 200:
            periods.append(200)
        
        for period in periods:
            if len(data) >= period:
                try:
                    # Simple Moving Average
                    data[f'SMA_{period}'] = data['Close'].rolling(
                        window=period, min_periods=max(1, period//4)
                    ).mean()
                    
                    # Exponential Moving Average
                    data[f'EMA_{period}'] = data['Close'].ewm(
                        span=period, adjust=False, min_periods=max(1, period//4)
                    ).mean()
                except Exception as e:
                    logger.warning(f"Failed to compute MA for period {period}: {e}")
                    continue
        
        # Crossovers (only if both MAs exist and have data)
        ma_pairs = [('SMA_20', 'SMA_50'), ('EMA_12', 'EMA_26')]
        for fast_ma, slow_ma in ma_pairs:
            if fast_ma in data.columns and slow_ma in data.columns:
                valid_mask = ~(data[fast_ma].isna() | data[slow_ma].isna())
                data[f'{fast_ma}_{slow_ma}_Crossover'] = 0
                data.loc[valid_mask, f'{fast_ma}_{slow_ma}_Crossover'] = np.where(
                    data.loc[valid_mask, fast_ma] > data.loc[valid_mask, slow_ma], 1, -1
                )
        
        return data
    
    def _compute_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Compute RSI with robust handling"""
        if len(data) < period:
            logger.warning(f"Insufficient data for RSI (need {period}, have {len(data)})")
            return data
        
        try:
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            
            # Handle division by zero
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            
            data['RSI_14'] = rsi.fillna(50)  # Neutral RSI value
            data['RSI_Signal'] = np.where(data['RSI_14'] > 70, -1, 
                                        np.where(data['RSI_14'] < 30, 1, 0))
            
            return data
        except Exception as e:
            logger.warning(f"RSI computation error: {e}")
            return data
    
    def _compute_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute MACD with error handling"""
        try:
            ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
            
            data['MACD'] = ema_12 - ema_26
            data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            
            # Crossover signal
            valid_mask = ~(data['MACD'].isna() | data['MACD_Signal'].isna())
            data['MACD_Crossover'] = 0
            data.loc[valid_mask, 'MACD_Crossover'] = np.where(
                data.loc[valid_mask, 'MACD'] > data.loc[valid_mask, 'MACD_Signal'], 1, -1
            )
            
            return data
        except Exception as e:
            logger.warning(f"MACD computation error: {e}")
            return data
    
    def _compute_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std: int = 2) -> pd.DataFrame:
        """Compute Bollinger Bands safely"""
        if len(data) < period:
            logger.warning(f"Insufficient data for Bollinger Bands (need {period})")
            return data
        
        try:
            rolling_mean = data['Close'].rolling(window=period, min_periods=max(1, period//4)).mean()
            rolling_std = data['Close'].rolling(window=period, min_periods=max(1, period//4)).std()
            
            data['BB_Middle'] = rolling_mean
            data['BB_Upper'] = rolling_mean + (rolling_std * std)
            data['BB_Lower'] = rolling_mean - (rolling_std * std)
            
            # Avoid division by zero
            bb_width = np.where(
                rolling_mean != 0,
                (data['BB_Upper'] - data['BB_Lower']) / rolling_mean,
                0
            )
            data['BB_Width'] = pd.Series(bb_width, index=data.index)
            
            # BB position (avoid division by zero)
            bb_range = data['BB_Upper'] - data['BB_Lower']
            bb_position = np.where(
                bb_range != 0,
                (data['Close'] - data['BB_Lower']) / bb_range,
                0.5
            )
            data['BB_Position'] = pd.Series(bb_position, index=data.index)
            
            return data
        except Exception as e:
            logger.warning(f"Bollinger Bands error: {e}")
            return data
    
    def _compute_obv(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute On-Balance Volume safely"""
        try:
            if 'Volume' not in data.columns:
                return data
            
            price_change = data['Close'].diff()
            volume_change = np.sign(price_change) * data['Volume']
            data['OBV'] = volume_change.fillna(0).cumsum()
            return data
        except Exception as e:
            logger.warning(f"OBV computation error: {e}")
            return data
    
    def _compute_atr(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Compute Average True Range safely"""
        try:
            if not all(col in data.columns for col in ['High', 'Low', 'Close']):
                logger.warning("Missing HLC data for ATR")
                return data
            
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            
            true_range = np.maximum.reduce([high_low, high_close, low_close])
            data['ATR_14'] = true_range.rolling(window=period, min_periods=1).mean()
            return data
        except Exception as e:
            logger.warning(f"ATR computation error: {e}")
            return data
    
    def _compute_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Compute Stochastic Oscillator safely"""
        try:
            if not all(col in data.columns for col in ['High', 'Low', 'Close']):
                logger.warning("Missing HLC data for Stochastic")
                return data
            
            if len(data) < k_period:
                logger.warning(f"Insufficient data for Stochastic (need {k_period})")
                return data
            
            lowest_low = data['Low'].rolling(window=k_period, min_periods=1).min()
            highest_high = data['High'].rolling(window=k_period, min_periods=1).max()
            
            # Avoid division by zero
            range_val = highest_high - lowest_low
            stoch_k = np.where(
                range_val != 0,
                100 * ((data['Close'] - lowest_low) / range_val),
                50
            )
            data['Stoch_K'] = pd.Series(stoch_k, index=data.index)
            data['Stoch_D'] = data['Stoch_K'].rolling(window=d_period, min_periods=1).mean()
            
            return data
        except Exception as e:
            logger.warning(f"Stochastic computation error: {e}")
            return data
    
    def _compute_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute volume-based indicators safely"""
        try:
            if 'Volume' not in data.columns:
                return data
            
            data['Volume_SMA_20'] = data['Volume'].rolling(window=20, min_periods=1).mean()
            
            # Avoid division by zero
            volume_sma = data['Volume_SMA_20'].replace(0, np.nan)
            volume_ratio = np.where(
                ~volume_sma.isna(),
                data['Volume'] / volume_sma,
                1.0
            )
            data['Volume_Ratio'] = pd.Series(volume_ratio, index=data.index)
            
            return data
        except Exception as e:
            logger.warning(f"Volume indicators error: {e}")
            return data
    
    def _compute_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute additional price features safely"""
        try:
            # Daily returns
            data['Daily_Return'] = data['Close'].pct_change().fillna(0)
            
            # Log returns
            log_return = np.log(data['Close'] / data['Close'].shift(1))
            data['Log_Return'] = log_return.fillna(0)
            
            # Price change features (if available)
            if 'Open' in data.columns:
                data['Price_Change'] = data['Close'] - data['Open']
                data['Close_Open_Ratio'] = data['Close'] / data['Open'].replace(0, np.nan)
                data['Close_Open_Ratio'] = data['Close_Open_Ratio'].fillna(1.0)
            
            if all(col in data.columns for col in ['High', 'Low']):
                data['High_Low_Ratio'] = data['High'] / data['Low'].replace(0, np.nan)
                data['High_Low_Ratio'] = data['High_Low_Ratio'].fillna(1.0)
            
            # Lag features with safety
            for lag in [1, 2, 3]:
                if len(data) > lag:
                    data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
                    if 'Volume' in data.columns:
                        data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
            
            return data
        except Exception as e:
            logger.warning(f"Price features error: {e}")
            return data
    
    def _compute_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility features safely"""
        try:
            if 'Daily_Return' not in data.columns:
                data['Daily_Return'] = data['Close'].pct_change().fillna(0)
            
            # Short-term volatility
            data['Volatility_5D'] = data['Daily_Return'].rolling(
                window=5, min_periods=1
            ).std().fillna(0)
            
            # Long-term volatility
            data['Volatility_20D'] = data['Daily_Return'].rolling(
                window=20, min_periods=5
            ).std().fillna(data['Volatility_5D'])
            
            # Volatility ratio
            vol_ratio = np.where(
                data['Volatility_20D'] != 0,
                data['Volatility_5D'] / data['Volatility_20D'],
                1.0
            )
            data['Volatility_Ratio'] = pd.Series(vol_ratio, index=data.index)
            
            return data
        except Exception as e:
            logger.warning(f"Volatility features error: {e}")
            return data
    
    def get_available_indicators(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Check which indicators can be computed"""
        available = {
            'price_based': ['Close'] in data.columns,
            'volume_based': ['Volume'] in data.columns,
            'hlc_based': all(col in data.columns for col in ['High', 'Low', 'Close'])
        }
        
        indicator_types = {
            'Moving Averages': available['price_based'],
            'RSI': available['price_based'],
            'MACD': available['price_based'],
            'Bollinger Bands': available['price_based'],
            'OBV': available['volume_based'],
            'ATR': available['hlc_based'],
            'Stochastic': available['hlc_based'],
            'Volume Indicators': available['volume_based']
        }
        
        return indicator_types
    
    def reset_indicators(self):
        """Reset indicator state"""
        self.indicators_computed = False
        self.timezone_handled = False
        self.original_data_shape = None