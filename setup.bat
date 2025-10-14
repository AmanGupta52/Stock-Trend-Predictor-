@echo off
chcp 65001 >nul
echo ===============================================
echo    📈 STOCK TREND PREDICTOR UI LAUNCHER
echo ===============================================
echo.
echo 🚀 Starting Stock Prediction UI...
echo.
echo 📊 If browser doesn't open automatically:
echo 🌐 Visit: http://localhost:8501
echo.
echo ⚠️  Press Ctrl+C in this window to stop the server
echo.
echo 🔄 Loading... Please wait...
echo.

python -m streamlit run stock_ui.py

echo.
echo ❌ Server stopped.
pause