#!/usr/bin/env python3
"""
Stock Market Trend Prediction System
Main entry point for the application
"""

import argparse
import logging
from src.data_fetcher import StockDataFetcher
from src.technical_indicators import TechnicalIndicators
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.predictor import StockPredictor
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_prediction.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_cli_mode(ticker: str, period: str = "1y", model_type: str = "random_forest"):
    """Run the system in CLI mode"""
    print(f"🚀 Starting Stock Prediction for {ticker}")
    
    try:
        # Fetch data
        fetcher = StockDataFetcher()
        data = fetcher.get_stock_data(ticker, period)
        print("✅ Data fetched successfully")
        
        # Compute indicators
        indicators = TechnicalIndicators()
        data = indicators.compute_all_indicators(data)
        print("✅ Technical indicators computed")
        
        # Prepare features
        engineer = FeatureEngineer()
        X, y = engineer.prepare_features(data)
        print("✅ Features prepared")
        
        # Train model
        trainer = ModelTrainer(model_type=model_type)
        performance = trainer.train_models(X, y, problem_type='classification')
        print("✅ Model trained")
        
        # Make prediction
        predictor = StockPredictor(trainer)
        latest_data = X.iloc[[-1]]
        prediction = predictor.predict_next_day(latest_data)
        
        print("\n" + "="*50)
        print("📊 PREDICTION RESULTS")
        print("="*50)
        print(f"Stock: {ticker}")
        print(f"Trend: {prediction['trend']}")
        print(f"Action: {prediction['action']}")
        print(f"Confidence: {prediction['confidence']:.2%}")
        print(f"Model: {prediction['model']}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error in CLI mode: {str(e)}")
        print(f"❌ Error: {str(e)}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Stock Market Trend Prediction System')
    parser.add_argument('--ticker', type=str, default='RELIANCE.NS', 
                       help='Stock ticker symbol')
    parser.add_argument('--period', type=str, default='1y', 
                       help='Data period (1y, 2y, 5y, etc.)')
    parser.add_argument('--model', type=str, default='random_forest',
                       choices=['random_forest', 'xgboost', 'logistic_regression'],
                       help='ML model to use')
    parser.add_argument('--mode', type=str, default='ui',
                       choices=['ui', 'cli'],
                       help='Run mode: ui (Streamlit) or cli')
    
    args = parser.parse_args()
    
    if args.mode == 'cli':
        run_cli_mode(args.ticker, args.period, args.model)
    else:
        print("Starting Streamlit UI...")
        print("Run: streamlit run stock_ui.py")
        # You would typically run Streamlit here
        # but it's better to run it directly

if __name__ == "__main__":
    main()