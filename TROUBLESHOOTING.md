# Stock Trend Predictor - Troubleshooting Guide

## 🚀 Quick Start
1. **Double-click** `run.bat` 
2. **Wait** for browser to open (may take 10-30 seconds)
3. **Use the app** at http://localhost:8501

## 🔧 Common Issues & Solutions

### ❌ "Port already in use"
```bash
# Kill existing Streamlit processes
taskkill /f /im python.exe
# Or use different port
python -m streamlit run stock_ui.py --server.port 8502