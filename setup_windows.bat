@echo off
echo Setting up Stock Prediction System on Windows...

:: Create virtual environment
python -m venv venv
echo Virtual environment created.

:: Activate virtual environment
call venv\Scripts\activate

:: Install dependencies
pip install --upgrade pip
pip install pytest streamlit yfinance pandas numpy scikit-learn xgboost plotly matplotlib seaborn joblib python-dotenv pyyaml requests streamlit-option-menu plotly-resampler

echo.
echo Setup completed!
echo.
echo To run the application:
echo venv\Scripts\activate
echo streamlit run stock_ui.py
echo.
echo To run tests:
echo python -m pytest tests/ -v
pause