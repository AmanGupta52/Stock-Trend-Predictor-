import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class StockVisualizer:
    def __init__(self, theme: str = 'light'):
        self.theme = theme
        self.set_theme(theme)
    
    def set_theme(self, theme: str):
        """Set visualization theme"""
        self.theme = theme
        if theme == 'dark':
            plt.style.use('dark_background')
            self.plot_bgcolor = 'rgba(0,0,0,0)'
            self.paper_bgcolor = 'rgba(0,0,0,0)'
            self.font_color = 'white'
        else:
            plt.style.use('default')
            self.plot_bgcolor = 'white'
            self.paper_bgcolor = 'white'
            self.font_color = 'black'
    
    def create_price_chart(self, data: pd.DataFrame, 
                          indicators: List[str] = None) -> go.Figure:
        """Create interactive price chart with indicators"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Price Chart', 'Volume'),
            row_width=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add moving averages if available
        if 'SMA_20' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['SMA_20'], 
                          line=dict(color='orange', width=1),
                          name='SMA 20'),
                row=1, col=1
            )
        
        if 'SMA_50' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['SMA_50'], 
                          line=dict(color='red', width=1),
                          name='SMA 50'),
                row=1, col=1
            )
        
        # Volume chart
        colors = ['red' if row['Open'] > row['Close'] else 'green' 
                 for _, row in data.iterrows()]
        
        fig.add_trace(
            go.Bar(x=data.index, y=data['Volume'], 
                  marker_color=colors, name='Volume'),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Stock Price and Volume',
            xaxis_rangeslider_visible=False,
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
            height=600
        )
        
        return fig
    
    def create_technical_indicators_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create chart with multiple technical indicators"""
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('RSI', 'MACD', 'Bollinger Bands', 'Stochastic'),
            row_heights=[0.25, 0.25, 0.25, 0.25]
        )
        
        # RSI
        if 'RSI_14' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['RSI_14'], 
                          line=dict(color='purple'), name='RSI'),
                row=1, col=1
            )
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
        
        # MACD
        if all(col in data.columns for col in ['MACD', 'MACD_Signal']):
            fig.add_trace(
                go.Scatter(x=data.index, y=data['MACD'], 
                          line=dict(color='blue'), name='MACD'),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['MACD_Signal'], 
                          line=dict(color='red'), name='Signal'),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(x=data.index, y=data['MACD_Histogram'], 
                      marker_color='gray', name='Histogram'),
                row=2, col=1
            )
        
        # Bollinger Bands
        if all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            fig.add_trace(
                go.Scatter(x=data.index, y=data['BB_Upper'], 
                          line=dict(color='gray', dash='dash'), 
                          name='BB Upper'),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['BB_Middle'], 
                          line=dict(color='blue'), name='BB Middle'),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['BB_Lower'], 
                          line=dict(color='gray', dash='dash'), 
                          name='BB Lower', fill='tonexty'),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['Close'], 
                          line=dict(color='orange'), name='Price'),
                row=3, col=1
            )
        
        # Stochastic
        if all(col in data.columns for col in ['Stoch_K', 'Stoch_D']):
            fig.add_trace(
                go.Scatter(x=data.index, y=data['Stoch_K'], 
                          line=dict(color='blue'), name='Stoch %K'),
                row=4, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['Stoch_D'], 
                          line=dict(color='red'), name='Stoch %D'),
                row=4, col=1
            )
            fig.add_hline(y=80, line_dash="dash", line_color="red", row=4, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="green", row=4, col=1)
        
        fig.update_layout(
            title='Technical Indicators',
            height=800,
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
            showlegend=True
        )
        
        return fig
    
    def create_prediction_chart(self, actual: pd.Series, 
                              predicted: pd.Series) -> go.Figure:
        """Create chart comparing actual vs predicted values"""
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(x=actual.index, y=actual.values,
                      mode='lines', name='Actual',
                      line=dict(color='blue'))
        )
        
        fig.add_trace(
            go.Scatter(x=predicted.index, y=predicted.values,
                      mode='lines', name='Predicted',
                      line=dict(color='red', dash='dash'))
        )
        
        fig.update_layout(
            title='Actual vs Predicted Prices',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
            height=400
        )
        
        return fig
    
    def create_feature_importance_chart(self, importance_df: pd.DataFrame, 
                                      top_n: int = 15) -> go.Figure:
        """Create feature importance bar chart"""
        top_features = importance_df.head(top_n)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title=f'Top {top_n} Feature Importance',
            color='importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
            height=500
        )
        
        return fig
    
    def create_correlation_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap"""
        # Select only numerical columns
        numerical_data = data.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numerical_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title='Correlation')
        ))
        
        fig.update_layout(
            title='Feature Correlation Heatmap',
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
            height=600
        )
        
        return fig
    
    def create_returns_distribution(self, returns: pd.Series) -> go.Figure:
        """Create distribution plot of returns"""
        fig = px.histogram(
            returns, 
            nbins=50,
            title='Daily Returns Distribution',
            marginal='box'
        )
        
        fig.update_layout(
            xaxis_title='Daily Returns',
            yaxis_title='Frequency',
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
            height=400
        )
        
        # Add mean line
        mean_return = returns.mean()
        fig.add_vline(x=mean_return, line_dash="dash", line_color="red")
        
        return fig