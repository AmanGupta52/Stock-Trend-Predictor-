import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import logging
from typing import Tuple, List, Optional
import warnings

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_columns = []
        self.original_index = None
    
    def prepare_features(self, data: pd.DataFrame, target_col: str = 'Next_Day_Close', 
                        problem_type: str = 'classification') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for model training with timezone handling"""
        try:
            logger.info("Starting feature engineering...")
            
            # **FIX 1: Handle timezone-aware timestamps**
            self._handle_timezone(data)
            
            # Create target variable
            data = self._create_target(data, problem_type)
            
            # Select and validate features
            feature_data = self._select_features(data)
            
            # Handle missing values robustly
            feature_data = self._handle_missing_values(feature_data)
            
            # Create additional features with safety checks
            feature_data = self._create_advanced_features(feature_data)
            
            # Align features and target
            X, y = self._align_features_target(feature_data, target_col, problem_type)
            
            # Final validation
            if X.empty or y.empty or len(X) < 10:
                raise ValueError(f"Insufficient data after processing: X={len(X)}, y={len(y)}")
            
            # Store feature columns
            self.feature_columns = X.columns.tolist()
            
            logger.info(f"✅ Prepared {len(X.columns)} features with {len(X)} samples")
            logger.info(f"Feature columns: {self.feature_columns[:5]}...")  # First 5 for logging
            
            return X, y
            
        except Exception as e:
            logger.error(f"❌ Feature engineering failed: {str(e)}")
            # Fallback to basic features
            logger.info("Attempting fallback feature engineering...")
            return self._create_fallback_features(data)
    
    def _handle_timezone(self, data: pd.DataFrame):
        """Handle timezone-aware timestamps"""
        if data.index.tz is not None:
            logger.info("Converting timezone-aware index to naive...")
            # Store original index for reference
            self.original_index = data.index
            # Convert to naive timestamps (remove timezone)
            data.index = data.index.tz_localize(None)
            logger.info("Timezone conversion completed")
    
    def _create_target(self, data: pd.DataFrame, problem_type: str) -> pd.DataFrame:
        """Create target variable with robust handling"""
        try:
            # Ensure we have enough data
            if len(data) < 5:
                raise ValueError("Insufficient data for target creation")
            
            # Next day closing price (regression)
            data['Next_Day_Close'] = data['Close'].shift(-1)
            
            # Calculate future returns safely
            future_returns = (data['Close'].shift(-1) / data['Close'] - 1).fillna(0)
            
            if problem_type == 'classification':
                # Binary classification: 1=UP, 0=DOWN
                data['Target_Binary'] = np.where(future_returns > 0, 1, 0)
                
                # Multi-class: strong down, down, up, strong up
                data['Target_Multi'] = pd.cut(
                    future_returns, 
                    bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf],
                    labels=[0, 1, 2, 3, 4]
                ).astype('Int64')  # Use nullable integer
            
            # Remove rows with NaN targets (last row always has NaN)
            data = data.dropna(subset=['Next_Day_Close', 'Target_Binary'])
            
            # Ensure proper data types
            if 'Target_Binary' in data.columns:
                data['Target_Binary'] = data['Target_Binary'].astype(int)
            
            logger.info(f"Created targets: {len(data)} valid samples")
            return data
            
        except Exception as e:
            logger.error(f"Target creation failed: {e}")
            # Fallback target
            data['Target_Binary'] = 0  # Default to HOLD
            return data
    
    def _select_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Select relevant features with validation"""
        try:
            # Base price columns (ensure they exist)
            base_cols = []
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col in data.columns:
                    base_cols.append(col)
                else:
                    logger.warning(f"Missing base column: {col}")
            
            # Technical indicator columns
            indicator_cols = []
            indicator_patterns = ['SMA_', 'EMA_', 'RSI', 'MACD', 'BB_', 'OBV', 'ATR', 'Stoch']
            for col in data.columns:
                if any(pattern in col for pattern in indicator_patterns):
                    indicator_cols.append(col)
            
            # Feature columns
            feature_cols = []
            feature_patterns = ['Return', 'Volatility', 'Ratio', 'Lag_', 'Crossover', 'Momentum']
            for col in data.columns:
                if any(pattern in col for pattern in feature_patterns):
                    feature_cols.append(col)
            
            all_features = base_cols + indicator_cols + feature_cols
            
            # Ensure we have basic features
            if not base_cols:
                logger.warning("No base columns found, using Close only")
                if 'Close' in data.columns:
                    all_features = ['Close']
                else:
                    raise ValueError("No price data available")
            
            # Select available columns + targets
            available_features = [col for col in all_features if col in data.columns]
            target_cols = ['Next_Day_Close', 'Target_Binary']
            target_cols = [col for col in target_cols if col in data.columns]
            
            selected_cols = available_features + target_cols
            feature_data = data[selected_cols].copy()
            
            logger.info(f"Selected {len(available_features)} features")
            return feature_data
            
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            # Fallback to minimal features
            return self._create_minimal_features(data)
    
    def _align_features_target(self, feature_data: pd.DataFrame, target_col: str, 
                              problem_type: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Ensure features and target are properly aligned"""
        try:
            # Determine target column
            if problem_type == 'classification':
                target_col = 'Target_Binary'
            elif target_col not in feature_data.columns:
                raise ValueError(f"Target column {target_col} not found")
            
            # Separate features and target
            if target_col in feature_data.columns:
                X = feature_data.drop(columns=[target_col], errors='ignore')
                y = feature_data[target_col]
            else:
                # Fallback
                X = feature_data.drop(columns=['Next_Day_Close'], errors='ignore')
                y = feature_data.get('Target_Binary', pd.Series(0, index=X.index))
            
            # Ensure same index
            common_index = X.index.intersection(y.index)
            if len(common_index) < 10:
                raise ValueError(f"Index alignment failed: only {len(common_index)} common indices")
            
            X = X.loc[common_index]
            y = y.loc[common_index]
            
            # Remove any remaining NaN
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            logger.info(f"Aligned X shape: {X.shape}, y shape: {len(y)}")
            return X, y
            
        except Exception as e:
            logger.error(f"Alignment failed: {e}")
            raise
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Robust missing value handling"""
        try:
            # Forward fill first (preserves time series order)
            data = data.ffill()
            
            # Backward fill remaining
            data = data.bfill()
            
            # Fill any remaining NaN with column means
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
            
            # Fill categorical with mode
            for col in data.select_dtypes(include=['object', 'category']).columns:
                data[col] = data[col].fillna(data[col].mode().iloc[0] if not data[col].mode().empty else 'Unknown')
            
            # Drop rows that still have NaN (should be minimal)
            initial_rows = len(data)
            data = data.dropna()
            final_rows = len(data)
            
            if final_rows < initial_rows * 0.8:  # More than 20% lost
                logger.warning(f"Significant data loss in missing value handling: {initial_rows} -> {final_rows}")
            
            logger.info(f"Missing values handled: {final_rows} rows remaining")
            return data
            
        except Exception as e:
            logger.error(f"Missing value handling failed: {e}")
            return data.dropna()
    
    def _create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features with safety checks"""
        try:
            df = data.copy()
            
            # Ensure we have Close column
            if 'Close' not in df.columns:
                logger.warning("No Close column for advanced features")
                return df
            
            # Price momentum (safe shifts)
            for days in [3, 5, 10]:
                safe_shift = df['Close'].shift(days)
                momentum = df['Close'] / safe_shift - 1
                df[f'Momentum_{days}D'] = momentum.fillna(0)
            
            # Daily returns
            df['Daily_Return'] = df['Close'].pct_change().fillna(0)
            
            # Volatility (rolling std)
            df['Volatility_5D'] = df['Daily_Return'].rolling(5, min_periods=1).std().fillna(0)
            df['Volatility_20D'] = df['Daily_Return'].rolling(20, min_periods=5).std().fillna(0)
            
            # Volume indicators
            if 'Volume' in df.columns:
                df['Volume_SMA_10'] = df['Volume'].rolling(10, min_periods=1).mean()
                df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_10'].replace(0, 1)
            
            # Price position indicators
            df['Price_Position'] = (df['Close'] - df['Close'].rolling(20, min_periods=5).min()) / \
                                 (df['Close'].rolling(20, min_periods=5).max() - df['Close'].rolling(20, min_periods=5).min())
            df['Price_Position'] = df['Price_Position'].fillna(0.5)
            
            # Seasonality (safe datetime operations)
            if isinstance(df.index, pd.DatetimeIndex):
                df['Day_of_Week'] = df.index.dayofweek
                df['Month'] = df.index.month
                df['Quarter'] = df.index.quarter
                df['Is_Month_End'] = df.index.is_month_end.astype(int)
            
            logger.info("Advanced features created successfully")
            return df
            
        except Exception as e:
            logger.warning(f"Advanced features failed: {e}. Using basic features.")
            return data
    
    def _create_fallback_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Create minimal fallback features"""
        logger.info("Creating fallback features...")
        
        try:
            if 'Close' not in data.columns:
                raise ValueError("No Close price data available")
            
            # Minimal features
            X = pd.DataFrame(index=data.index)
            X['Close'] = data['Close']
            X['Returns'] = data['Close'].pct_change().fillna(0)
            
            if 'Volume' in data.columns:
                X['Volume'] = data['Volume']
                X['Volume_Change'] = data['Volume'].pct_change().fillna(0)
            
            # Simple target
            y = (data['Close'].shift(-1) > data['Close']).astype(int).fillna(0)
            
            # Align
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]
            
            X = X.fillna(0)
            logger.info(f"Fallback features created: {X.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Fallback features failed: {e}")
            raise ValueError("Unable to create any valid features")
    
    def _create_minimal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create absolute minimal features"""
        available_cols = [col for col in ['Close', 'Volume', 'Open'] if col in data.columns]
        if not available_cols:
            raise ValueError("No basic price/volume data available")
        
        return data[available_cols].copy()
    
    def scale_features(self, X_train: pd.DataFrame, X_test: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Scale features with robust index preservation"""
        try:
            # Ensure numeric data
            X_train_numeric = X_train.select_dtypes(include=[np.number])
            
            if X_train_numeric.empty:
                logger.warning("No numeric features for scaling, returning raw data")
                return X_train, X_test
            
            # Fit scaler on training data only
            self.scaler.fit(X_train_numeric)
            
            # Transform training data
            X_train_scaled = pd.DataFrame(
                self.scaler.transform(X_train_numeric),
                columns=X_train_numeric.columns,
                index=X_train_numeric.index
            )
            
            # Preserve non-numeric columns
            non_numeric_cols = set(X_train.columns) - set(X_train_numeric.columns)
            for col in non_numeric_cols:
                X_train_scaled[col] = X_train[col]
            
            if X_test is not None:
                X_test_numeric = X_test.select_dtypes(include=[np.number])
                X_test_scaled = pd.DataFrame(
                    self.scaler.transform(X_test_numeric),
                    columns=X_test_numeric.columns,
                    index=X_test_numeric.index
                )
                
                # Preserve non-numeric columns
                non_numeric_cols_test = set(X_test.columns) - set(X_test_numeric.columns)
                for col in non_numeric_cols_test:
                    X_test_scaled[col] = X_test[col]
                
                return X_train_scaled, X_test_scaled
            else:
                return X_train_scaled, None
                
        except Exception as e:
            logger.warning(f"Scaling failed: {e}. Returning unscaled features.")
            return X_train, X_test
    
    def get_feature_importance_names(self) -> List[str]:
        """Get names of engineered features"""
        return self.feature_columns[:20]  # Return top features for visualization
    
    def inverse_transform(self, X_scaled: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform scaled features"""
        try:
            numeric_cols = X_scaled.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                X_numeric_scaled = X_scaled[numeric_cols]
                X_original = self.scaler.inverse_transform(X_numeric_scaled)
                X_result = pd.DataFrame(X_original, columns=numeric_cols, index=X_numeric_scaled.index)
                
                # Preserve non-numeric
                for col in set(X_scaled.columns) - set(numeric_cols):
                    X_result[col] = X_scaled[col]
                return X_result
            return X_scaled
        except:
            return X_scaled