📈 Indian Stock Trend Predictor
A comprehensive machine learning pipeline for predicting Indian stock price movements with both command-line interface and web UI.

https://img.shields.io/badge/Stock-Predictor-blue
https://img.shields.io/badge/Python-3.8%252B-green
https://img.shields.io/badge/ML-Random%2520Forest%252BXGBoost-orange

🎯 Overview
This project provides a complete solution for predicting Indian stock price trends using machine learning. It includes:

🤖 ML Pipeline: Random Forest and XGBoost models for stock prediction

📊 Technical Analysis: 20+ technical indicators

🌐 Web UI: Streamlit-based interactive interface

🔮 Predictions: Next-day trend forecasts with probabilities

📈 Visualization: Comprehensive charts and analysis

✨ Features
Core Features
Multi-Model Approach: Random Forest + XGBoost ensemble

Technical Indicators: RSI, MACD, Bollinger Bands, Moving Averages, etc.

Time-Series Validation: Proper time-based train/test split

Feature Importance: Identify most predictive indicators

Performance Metrics: ROC AUC, Classification Reports, Confusion Matrices

UI Features
Real-time Data: Live stock data from Yahoo Finance

Interactive Charts: Multiple technical analysis views

Simple Predictions: Rule-based buy/sell signals

Mobile Friendly: Responsive design

Data Export: Download processed datasets

🚀 Quick Start
Prerequisites
Python 3.8 or higher

pip (Python package manager)

Installation
Clone or download the project files

Install dependencies:

bash
# For command-line version
pip install -r requirements.txt

# For UI version
pip install -r requirements_ui.txt
File Structure
text
stock_trend_predictor/
├── stock_trend_pipeline_indian.py    # Main ML pipeline
├── stock_ui.py                       # Streamlit UI
├── requirements.txt                  # CLI dependencies
├── requirements_ui.txt              # UI dependencies
└── README.md                        # This file
📖 Usage
Method 1: Command Line Interface
Run the main script:

bash
python stock_trend_pipeline_indian.py
Enter stock name when prompted:

text
Enter stock symbol (e.g., RELIANCE, TCS, INFY): RELIANCE
View results:

Model performance metrics

Feature importance charts

Next-day predictions

Backtesting results

Method 2: Web Interface
Launch the UI:

bash
streamlit run stock_ui.py
Open your browser to http://localhost:8501

Use the sidebar to:

Enter stock name

Adjust date range

Set prediction horizon

Configure threshold

🏢 Supported Stocks
Popular Indian Stocks
Stock	Symbol	Description
Reliance Industries	RELIANCE	Conglomerate
Tata Consultancy	TCS	IT Services
Infosys	INFY	IT Services
HDFC Bank	HDFC BANK	Banking
ICICI Bank	ICICI BANK	Banking
State Bank of India	SBI	Banking
Hindustan Unilever	HUL	FMCG
ITC Limited	ITC	Conglomerate
Max Healthcare	MAX HEALTHCARE	Healthcare
Bharti Airtel	BHARTI AIRTEL	Telecom
Stock Name Format
Use short names: RELIANCE, TCS, INFY

For multi-word: HDFC BANK, MAX HEALTHCARE

The system automatically adds .NS suffix for NSE stocks

🔧 Technical Details
Machine Learning Models
1. Random Forest Classifier
Estimators: 100-200 trees

Max Depth: 6 levels

Features: 20+ technical indicators

Advantages: Robust to overfitting, handles non-linear relationships

2. XGBoost Classifier
Gradient Boosting: Sequential tree building

Regularization: Prevents overfitting

Optimized: For performance and accuracy

Advantages: High accuracy, feature importance

Technical Indicators
Price-based Indicators
Moving Averages: SMA 5, 10, EMA 10, 21

Bollinger Bands: Upper, Lower, Width

RSI: 14-period Relative Strength Index

MACD: Moving Average Convergence Divergence

ATR: 14-period Average True Range

Return & Momentum
Daily Returns: 1-day, 5-day percentage changes

Momentum: 7-day price momentum

Lag Features: Previous 1,2,3,5 day closes

Volume Indicators
Volume MA: 5-day, 21-day moving averages

Volume Ratio: Current vs average volume

Feature Engineering Pipeline
Data Download → Yahoo Finance API

Indicator Calculation → 20+ technical features

Label Creation → Binary classification (UP/DOWN)

Scaling → StandardScaler for normalization

Time-Series Split → 80% train, 20% test

Model Training → RF + XGBoost

Evaluation → Multiple metrics

Prediction → Next-day forecast

📊 Output & Results
Model Evaluation Metrics
Classification Report: Precision, Recall, F1-Score

ROC AUC: Area Under ROC Curve

Confusion Matrix: True/False Positives/Negatives

Visualizations
Feature Importance - Top 15 most important indicators

Confusion Matrices - Model performance comparison

Cumulative Returns - Strategy vs Buy & Hold

Technical Charts - Price and indicator plots

Prediction Output
text
🎯 NEXT DAY PREDICTION for RELIANCE.NS:
   📅 Date: 2024-01-15
   💰 Current Price: ₹2,845.50
   📈 Direction: UP
   📊 Probability UP: 68.42%
   📉 Probability DOWN: 31.58%
   💚 STRONG BUY SIGNAL
⚙️ Configuration
Command Line Parameters
python
ticker='RELIANCE'      # Stock symbol
start='2018-01-01'    # Start date
end=None              # End date (current if None)
horizon=1             # Prediction horizon in days
threshold=0.0         # Minimum return threshold
UI Configuration Options
Date Range: Custom start and end dates

Prediction Horizon: 1 to 7 days ahead

Return Threshold: 0% to 5% minimum return

Technical Views: Multiple chart types

🛠️ Development
Project Structure
python
# Core Functions
download_indian_stock_data()    # Data acquisition
add_technical_indicators()      # Feature engineering
create_labels()                 # Target variable creation
build_and_evaluate_indian_stock() # Main pipeline

# UI Components
analyze_stock()                 # UI analysis function
technical_charts()              # Visualization functions
prediction_display()            # Results formatting
Adding New Indicators
python
def add_custom_indicators(df):
    # Add your custom indicators here
    df['custom_indicator'] = df['close'].rolling(10).mean()
    return df
Model Customization
python
# Modify model parameters in build_and_evaluate_indian_stock()
rf = RandomForestClassifier(
    n_estimators=200,    # Increase trees
    max_depth=10,        # Increase depth
    random_state=42
)
📈 Performance Tips
For Better Accuracy
More Data: Use longer time periods (3+ years)

Feature Engineering: Add domain-specific indicators

Hyperparameter Tuning: Use RandomizedSearchCV

Ensemble Methods: Combine multiple model predictions

For Faster Execution
Reduce Features: Use top N important features only

Smaller Models: Reduce n_estimators for faster training

Caching: Save and reload trained models

Parallel Processing: Use n_jobs=-1 for multi-core

🐛 Troubleshooting
Common Issues
"No data found for symbol"

Check stock symbol spelling

Try alternative names (e.g., "HDFC" instead of "HDFC BANK")

Verify the stock trades on NSE

"Not enough data after cleaning"

Use longer time period

Reduce number of technical indicators

Adjust start date to earlier period

Model performance issues

Increase training data

Try different prediction horizons

Adjust return threshold

Memory errors

Reduce dataset size

Use fewer technical indicators

Clear cached models

Debug Mode
Add this at the top of your script for detailed logs:

python
import logging
logging.basicConfig(level=logging.DEBUG)
🔮 Future Enhancements
Planned Features
Sentiment Analysis: News and social media integration

Multiple Timeframes: Intraday, weekly, monthly analysis

Portfolio Optimization: Multi-stock recommendations

Advanced Models: LSTM, Transformer networks

Backtesting Engine: Comprehensive strategy testing

Alert System: Email/ SMS notifications

Research Directions
Alternative Data: Options flow, institutional activity

Market Regime Detection: Bull/Bear market adaptation

Risk Management: Position sizing, stop-loss optimization

Explainable AI: Model interpretation and reasoning

📚 Educational Resources
Stock Market Basics
Technical Analysis vs Fundamental Analysis

Risk Management Principles

Market Psychology

Machine Learning Concepts
Time Series Forecasting

Feature Engineering

Model Evaluation Metrics

Overfitting Prevention

Python Libraries
yfinance: Stock data download

scikit-learn: Machine learning algorithms

XGBoost: Gradient boosting implementation

Streamlit: Web application framework

🤝 Contributing
We welcome contributions! Please:

Fork the repository

Create a feature branch

Add tests for new functionality

Submit a pull request

Areas for Contribution
New technical indicators

Additional machine learning models

UI/UX improvements

Documentation enhancements

Bug fixes and optimizations

⚠️ Disclaimer
Important: This software is for educational and research purposes only.

📊 Not Financial Advice: Predictions are based on historical patterns and should not be considered investment advice

⚠️ High Risk: Stock market investments carry significant risk of loss

🔍 Backtested Performance: Past performance does not guarantee future results

💼 Professional Advice: Consult with qualified financial advisors before making investment decisions

The developers are not responsible for any financial losses incurred through the use of this software.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🆘 Support
Getting Help
📖 Documentation: Check this README first

🐛 Issues: Use GitHub issues for bug reports

💬 Discussions: Join the community forum

📧 Contact: Reach out to the development team

Resources
Yahoo Finance API Documentation

Scikit-learn Documentation

XGBoost Documentation

Streamlit Documentation

🎉 Acknowledgments
Yahoo Finance for providing free stock data

Scikit-learn and XGBoost teams for excellent ML libraries

Streamlit for making web apps accessible

Contributors and Testers who helped improve this project

Happy Predicting! 🚀📈