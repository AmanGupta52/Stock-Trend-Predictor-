@echo off
title Stock Trend Predictor
color 0A
echo.
echo ###############################################
echo #          STOCK TREND PREDICTOR             #
echo ###############################################
echo.
echo Initializing AI Stock Analysis System...
timeout /t 2 /nobreak >nul
echo.
echo 📊 Loading Machine Learning Models...
timeout /t 1 /nobreak >nul
echo 📈 Preparing Technical Indicators...
timeout /t 1 /nobreak >nul
echo 🔮 Initializing Prediction Engine...
timeout /t 1 /nobreak >nul
echo.
echo 🚀 LAUNCHING APPLICATION...
echo.
echo 💡 The app will open in your browser at:
echo    http://localhost:8501
echo.
echo ⚠️  Keep this window open while using the app
echo.
timeout /t 3 /nobreak >nul

python -m streamlit run stock_ui.py

pause