# stock_ui.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Stock Trend Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid;
    }
    .buy-signal {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .sell-signal {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    .hold-signal {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">📈 Indian Stock Trend Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
This app predicts the next day's stock price direction using Machine Learning models (Random Forest & XGBoost).
Enter any Indian stock name and get instant predictions with technical analysis.
""")

# Sidebar for input and information
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Stock input
    stock_name = st.text_input(
        "Enter Stock Name", 
        value="RELIANCE",
        help="Enter Indian stock name like: RELIANCE, TCS, INFY, HDFC BANK, etc."
    )
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2018, 1, 1),
            max_value=datetime.now() - timedelta(days=30)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now()
        )
    
    # Prediction horizon
    horizon = st.selectbox(
        "Prediction Horizon",
        options=[1, 2, 3, 5, 7],
        index=0,
        help="Number of days ahead to predict"
    )
    
    # Threshold for prediction
    threshold = st.slider(
        "Minimum Return Threshold (%)",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.1,
        help="Minimum percentage return to consider as 'UP' movement"
    )
    
    # Analyze button
    analyze_btn = st.button("🚀 Analyze Stock", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.header("📋 Popular Stocks")
    popular_stocks = ["RELIANCE", "TCS", "INFY", "HDFC BANK", "ICICI BANK", "SBI", "HUL", "ITC"]
    for stock in popular_stocks:
        if st.button(f"📊 {stock}", use_container_width=True):
            stock_name = stock
            st.rerun()
    
    st.markdown("---")
    st.info("""
    **💡 Tips:**
    - Use short names: 'RELIANCE', 'TCS'
    - For banks: 'HDFC BANK', 'ICICI BANK'
    - Multi-word: 'MAX HEALTHCARE'
    """)

# Technical indicators function (same as your original)
def add_technical_indicators(df):
    """
    Adds common technical indicators to a stock price DataFrame.
    """
    if df is None or df.empty:
        return df
        
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    try:
        # Moving Averages
        df['sma_5'] = close.rolling(window=5).mean()
        df['sma_10'] = close.rolling(window=10).mean()
        df['ema_10'] = close.ewm(span=10, adjust=False).mean()
        df['ema_21'] = close.ewm(span=21, adjust=False).mean()

        # Returns & Momentum
        df['ret_1'] = close.pct_change(1)
        df['ret_5'] = close.pct_change(5)
        df['momentum_7'] = close - close.shift(7)

        # Bollinger Bands
        sma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        df['bb_upper'] = sma20 + 2 * std20
        df['bb_lower'] = sma20 - 2 * std20
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma20

        # RSI
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.rolling(window=14).mean()
        roll_down = down.rolling(window=14).mean()
        rs = roll_up / roll_down
        df['rsi_14'] = 100.0 - (100.0 / (1.0 + rs))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # ATR
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(window=14).mean()

        # Volume Features
        df['vol_5'] = volume.rolling(window=5).mean()
        df['vol_21'] = volume.rolling(window=21).mean()
        df['vol_ratio'] = volume / (df['vol_21'] + 1e-9)

        # Lag Features
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = close.shift(lag)

    except Exception as e:
        st.error(f"Error adding technical indicators: {e}")
    
    return df

def create_labels(df, horizon=1, threshold=0.0):
    """
    Create target labels for prediction
    """
    try:
        future_close = df['close'].shift(-horizon)
        df['future_return'] = (future_close - df['close']) / df['close']
        df['target'] = (df['future_return'] > threshold/100).astype(int)
    except Exception as e:
        st.error(f"Error creating labels: {e}")
    
    return df

def download_stock_data(ticker, start_date, end_date):
    """
    Download stock data with error handling
    """
    try:
        # Clean ticker name
        ticker = ticker.upper().strip()
        words_to_remove = ['LIMITED', 'LTD', 'INC', 'CORPORATION', 'CO', 'AND', 'THE']
        ticker_parts = [word for word in ticker.split() if word not in words_to_remove]
        
        if len(ticker_parts) > 3:
            clean_ticker = ''.join(ticker_parts[:3])
        else:
            clean_ticker = ''.join(ticker_parts)
        
        # Add .NS suffix
        if not clean_ticker.endswith('.NS'):
            yf_ticker = clean_ticker + '.NS'
        else:
            yf_ticker = clean_ticker
        
        st.info(f"📥 Downloading data for {yf_ticker}...")
        
        df = yf.download(yf_ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            # Try alternative
            alt_ticker = clean_ticker.replace(' ', '') + '.NS'
            if alt_ticker != yf_ticker:
                st.info(f"Trying alternative: {alt_ticker}")
                df = yf.download(alt_ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            st.error(f"❌ No data found for {ticker}")
            return None, None
            
        # Handle columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [str(col).lower() for col in df.columns]
        
        df.index = pd.to_datetime(df.index)
        return df, yf_ticker
        
    except Exception as e:
        st.error(f"❌ Download error: {e}")
        return None, None

# Main analysis function
def analyze_stock(stock_name, start_date, end_date, horizon, threshold):
    """
    Main analysis function for the UI
    """
    # Download data
    with st.spinner("Downloading stock data..."):
        df, actual_ticker = download_stock_data(stock_name, start_date, end_date)
    
    if df is None:
        return None
    
    # Display basic info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = df['close'].iloc[-1]
        st.metric("Current Price", f"₹{current_price:.2f}")
    
    with col2:
        price_change = df['close'].iloc[-1] - df['close'].iloc[-2]
        change_pct = (price_change / df['close'].iloc[-2]) * 100
        st.metric("Daily Change", f"₹{price_change:.2f}", f"{change_pct:.2f}%")
    
    with col3:
        volume = df['volume'].iloc[-1]
        st.metric("Volume", f"{volume:,.0f}")
    
    with col4:
        days_data = len(df)
        st.metric("Data Points", f"{days_data}")
    
    # Price chart
    st.subheader("📊 Price Chart")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['close'], label='Close Price', linewidth=2)
    ax.set_title(f'{actual_ticker} - Stock Price', fontsize=16)
    ax.set_ylabel('Price (₹)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    
    # Add technical indicators
    with st.spinner("Calculating technical indicators..."):
        df = add_technical_indicators(df)
        df = create_labels(df, horizon, threshold)
        df = df.dropna().copy()
    
    if len(df) < 100:
        st.error("❌ Not enough data after processing. Try a longer time period.")
        return None
    
    # Technical Analysis
    st.subheader("🔧 Technical Analysis")
    
    # Create tabs for different technical views
    tab1, tab2, tab3, tab4 = st.tabs(["Moving Averages", "RSI & Momentum", "Bollinger Bands", "MACD"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['close'], label='Close Price', linewidth=2)
        ax.plot(df.index, df['sma_5'], label='SMA 5', alpha=0.7)
        ax.plot(df.index, df['sma_10'], label='SMA 10', alpha=0.7)
        ax.plot(df.index, df['ema_21'], label='EMA 21', alpha=0.7)
        ax.set_title('Moving Averages')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with tab2:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # RSI
        ax1.plot(df.index, df['rsi_14'], label='RSI 14', color='purple', linewidth=2)
        ax1.axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax1.axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax1.set_title('RSI Indicator')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Momentum
        ax2.plot(df.index, df['momentum_7'], label='7-Day Momentum', color='orange', linewidth=2)
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Price Momentum')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab3:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['close'], label='Close Price', linewidth=2)
        ax.plot(df.index, df['bb_upper'], label='BB Upper', alpha=0.7, color='red')
        ax.plot(df.index, df['bb_lower'], label='BB Lower', alpha=0.7, color='green')
        ax.fill_between(df.index, df['bb_upper'], df['bb_lower'], alpha=0.2, color='gray')
        ax.set_title('Bollinger Bands')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with tab4:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # MACD
        ax1.plot(df.index, df['macd'], label='MACD', linewidth=2)
        ax1.plot(df.index, df['macd_signal'], label='Signal', linewidth=2)
        ax1.bar(df.index, df['macd_hist'], label='Histogram', alpha=0.5)
        ax1.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax1.set_title('MACD')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volume
        ax2.bar(df.index, df['volume'], alpha=0.7, color='blue', label='Volume')
        ax2.plot(df.index, df['vol_5'], color='red', linewidth=2, label='5-day Avg Volume')
        ax2.set_title('Volume')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Current technical values
    st.subheader("📈 Current Technical Values")
    
    tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
    
    with tech_col1:
        current_rsi = df['rsi_14'].iloc[-1]
        rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
        rsi_color = "red" if current_rsi > 70 else "green" if current_rsi < 30 else "orange"
        st.metric("RSI (14)", f"{current_rsi:.1f}", rsi_status, delta_color="off")
    
    with tech_col2:
        macd_val = df['macd'].iloc[-1]
        macd_signal = df['macd_signal'].iloc[-1]
        macd_trend = "Bullish" if macd_val > macd_signal else "Bearish"
        st.metric("MACD", f"{macd_val:.2f}", macd_trend, delta_color="off")
    
    with tech_col3:
        bb_position = (df['close'].iloc[-1] - df['bb_lower'].iloc[-1]) / (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1])
        bb_status = "Upper Band" if bb_position > 0.8 else "Lower Band" if bb_position < 0.2 else "Middle"
        st.metric("Bollinger Position", f"{bb_position:.1%}", bb_status, delta_color="off")
    
    with tech_col4:
        volume_ratio = df['vol_ratio'].iloc[-1]
        volume_status = "High" if volume_ratio > 1.5 else "Low" if volume_ratio < 0.5 else "Normal"
        st.metric("Volume Ratio", f"{volume_ratio:.2f}", volume_status, delta_color="off")
    
    # Simple prediction based on technicals
    st.subheader("🎯 Simple Technical Prediction")
    
    # Simple rule-based prediction
    buy_signals = 0
    sell_signals = 0
    
    if current_rsi < 40:
        buy_signals += 1
    elif current_rsi > 60:
        sell_signals += 1
    
    if macd_val > macd_signal:
        buy_signals += 1
    else:
        sell_signals += 1
    
    if bb_position < 0.3:
        buy_signals += 1
    elif bb_position > 0.7:
        sell_signals += 1
    
    if volume_ratio > 1.2:
        buy_signals += 1
    
    total_signals = buy_signals + sell_signals
    if total_signals > 0:
        buy_confidence = buy_signals / total_signals
    else:
        buy_confidence = 0.5
    
    # Display prediction
    if buy_confidence > 0.6:
        signal_class = "buy-signal"
        signal_text = "STRONG BUY SIGNAL"
        signal_emoji = "💚"
    elif buy_confidence > 0.4:
        signal_class = "hold-signal"
        signal_text = "HOLD SIGNAL"
        signal_emoji = "💛"
    else:
        signal_class = "sell-signal"
        signal_text = "SELL SIGNAL"
        signal_emoji = "❤️"
    
    st.markdown(f"""
    <div class="prediction-box {signal_class}">
        <h3>{signal_emoji} {signal_text}</h3>
        <p><strong>Confidence:</strong> {buy_confidence:.1%}</p>
        <p><strong>Buy Signals:</strong> {buy_signals} | <strong>Sell Signals:</strong> {sell_signals}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Explanation
    with st.expander("📖 Technical Analysis Explanation"):
        st.markdown("""
        **Technical Indicators Used:**
        
        - **RSI (Relative Strength Index)**: Measures momentum. Below 30 = Oversold (Buy), Above 70 = Overbought (Sell)
        - **MACD**: Trend-following momentum indicator. MACD > Signal = Bullish
        - **Bollinger Bands**: Price near lower band = Potential buy, near upper band = Potential sell
        - **Volume Ratio**: High volume confirms price movements
        
        **Note:** This is a simplified technical analysis. For comprehensive analysis, use the machine learning models from the main script.
        """)
    
    return df, actual_ticker

# Main app logic
if analyze_btn and stock_name:
    with st.container():
        result = analyze_stock(stock_name, start_date, end_date, horizon, threshold)
        
        if result:
            df, actual_ticker = result
            st.success(f"✅ Analysis completed for {actual_ticker}!")
            
            # Additional information
            st.subheader("📋 Dataset Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**First 5 rows:**")
                st.dataframe(df.head()[['open', 'high', 'low', 'close', 'volume']].style.format("{:.2f}"))
            
            with col2:
                st.write("**Statistical Summary:**")
                st.dataframe(df[['open', 'high', 'low', 'close', 'volume']].describe().style.format("{:.2f}"))
            
            # Download data option
            csv = df.to_csv()
            st.download_button(
                label="📥 Download Processed Data",
                data=csv,
                file_name=f"{actual_ticker}_technical_data.csv",
                mime="text/csv"
            )

else:
    # Welcome screen
    st.markdown("""
    ## 🎯 Welcome to Stock Trend Predictor!
    
    This application provides:
    
    - **📊 Real-time stock data** for Indian companies
    - **🔧 Technical analysis** with multiple indicators
    - **📈 Interactive charts** for better visualization
    - **🎯 Simple predictions** based on technical signals
    
    ### 🚀 Getting Started:
    
    1. **Enter a stock name** in the sidebar (e.g., RELIANCE, TCS, HDFC BANK)
    2. **Adjust the date range** if needed
    3. **Click 'Analyze Stock'** to run the analysis
    4. **View results** in the main panel
    
    ### 📋 Popular Stocks to Try:
    - **RELIANCE** - Reliance Industries
    - **TCS** - Tata Consultancy Services
    - **INFY** - Infosys
    - **HDFC BANK** - HDFC Bank Limited
    - **ICICI BANK** - ICICI Bank
    
    *Note: For machine learning model predictions with Random Forest and XGBoost, use the main script.*
    """)
    
    # Sample chart
    st.subheader("📈 Sample Analysis - RELIANCE")
    try:
        sample_df, _ = download_stock_data("RELIANCE", 
                                         datetime(2023, 1, 1), 
                                         datetime.now())
        if sample_df is not None:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(sample_df.index, sample_df['close'], linewidth=2)
            ax.set_title("RELIANCE - Sample Price Chart")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    except:
        pass

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Stock Trend Predictor | Built with Streamlit | Data from Yahoo Finance"
    "</div>", 
    unsafe_allow_html=True
)