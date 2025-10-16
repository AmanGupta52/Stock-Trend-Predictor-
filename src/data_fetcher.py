import yfinance as yf
import pandas as pd
import requests
import logging
from datetime import datetime, timedelta
import time
import re
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class StockDataFetcher:
    def __init__(self, cache_duration: int = 3600):
        self.cache_duration = cache_duration
        self._cache = {}
        self.symbol_mappings = self._load_symbol_mappings()
    
    def _load_symbol_mappings(self) -> Dict[str, str]:
        """Load comprehensive symbol mappings"""
        return {
            # BLS International
            'BLS': 'BLS.NS',
            'BLS INTERNATIONAL': 'BLS.NS',
            'BLS INTERNATIONAL SERVICES': 'BLS.NS',
            'BLS INTL': 'BLS.NS',
            
            # Major Nifty 50 stocks
            'RELIANCE': 'RELIANCE.NS',
            'TCS': 'TCS.NS', 
            'INFY': 'INFY.NS',
            'INFOSYS': 'INFY.NS',
            'HDFC': 'HDFCBANK.NS',
            'HDFC BANK': 'HDFCBANK.NS',
            'HDFCBANK': 'HDFCBANK.NS',
            'HUL': 'HINDUNILVR.NS',
            'HINDUSTAN UNILEVER': 'HINDUNILVR.NS',
            'UNILEVER': 'HINDUNILVR.NS',
            
            # Adani Group
            'ADANI POWER': 'ADANIPOWER.NS',
            'ADANIPOWER': 'ADANIPOWER.NS',
            'ADANI PORTS': 'ADANIPORTS.NS',
            'ADANIPORTS': 'ADANIPORTS.NS',
            'ADANI ENTERPRISES': 'ADANIENT.NS',
            'ADANIENT': 'ADANIENT.NS',
            'ADANI GREEN': 'ADANIGREEN.NS',
            'ADANIGREEN': 'ADANIGREEN.NS',
            'ADANI TRANSMISSION': 'ADANITRANS.NS',
            'ADANITRANS': 'ADANITRANS.NS',
            'ADANI TOTAL GAS': 'ADANIGAS.NS',
            'ADANIGAS': 'ADANIGAS.NS',
            'ADANI WILMAR': 'AWL.NS',
            'AWL': 'AWL.NS',
            
            # Banks
            'SBIN': 'SBIN.NS',
            'STATE BANK': 'SBIN.NS',
            'STATE BANK OF INDIA': 'SBIN.NS',
            'ICICI': 'ICICIBANK.NS',
            'ICICI BANK': 'ICICIBANK.NS',
            'ICICIBANK': 'ICICIBANK.NS',
            'AXIS BANK': 'AXISBANK.NS',
            'AXISBANK': 'AXISBANK.NS',
            'KOTAK BANK': 'KOTAKBANK.NS',
            'KOTAKBANK': 'KOTAKBANK.NS',
            'BAJFINANCE': 'BAJFINANCE.NS',
            'BAJAJ FINANCE': 'BAJFINANCE.NS',
            
            # Others
            'BHARTI AIRTEL': 'BHARTIARTL.NS',
            'AIRTEL': 'BHARTIARTL.NS',
            'BHARTIARTL': 'BHARTIARTL.NS',
            'ITC': 'ITC.NS',
            'LT': 'LT.NS',
            'LARSEN': 'LT.NS',
            'LARSEN TOUBRO': 'LT.NS',
            'L&T': 'LT.NS',
            'ASIAN PAINTS': 'ASIANPAINT.NS',
            'ASIANPAINT': 'ASIANPAINT.NS',
            'MARUTI': 'MARUTI.NS',
            'MARUTI SUZUKI': 'MARUTI.NS',
            'TATA MOTORS': 'TATAMOTORS.NS',
            'TATAMOTORS': 'TATAMOTORS.NS',
            'TATA STEEL': 'TATASTEEL.NS',
            'TATASTEEL': 'TATASTEEL.NS',
            'WIPRO': 'WIPRO.NS',
            'ONGC': 'ONGC.NS',
            'NTPC': 'NTPC.NS',
            'POWERGRID': 'POWERGRID.NS',
            'POWER GRID': 'POWERGRID.NS',
            'COAL INDIA': 'COALINDIA.NS',
            'COALINDIA': 'COALINDIA.NS',
            'SUN PHARMA': 'SUNPHARMA.NS',
            'SUNPHARMA': 'SUNPHARMA.NS',
            'DR REDDY': 'DRREDDY.NS',
            'DRREDDY': 'DRREDDY.NS',
            'CIPLA': 'CIPLA.NS',
            'TECH MAHINDRA': 'TECHM.NS',
            'TECHM': 'TECHM.NS',
            'HCLTECH': 'HCLTECH.NS',
            'HCL TECH': 'HCLTECH.NS',
            'ULTRACEMCO': 'ULTRACEMCO.NS',
            'ULTRACEM': 'ULTRACEMCO.NS',
            'NESTLE': 'NESTLEIND.NS',
            'NESTLE INDIA': 'NESTLEIND.NS',
            'TITAN': 'TITAN.NS',
            'BRITANNIA': 'BRITANNIA.NS',
            'BAJAJFINSV': 'BAJAJFINSV.NS',
            'BAJAJ FINSERV': 'BAJAJFINSV.NS',
            'GRASIM': 'GRASIM.NS',
            'HEROMOTOCO': 'HEROMOTOCO.NS',
            'HERO': 'HEROMOTOCO.NS',
            'DIVISLAB': 'DIVISLAB.NS',
            'DIVI': 'DIVISLAB.NS',
            'EICHERMOT': 'EICHERMOT.NS',
            'EICHER': 'EICHERMOT.NS',
            'SHRIRAMFIN': 'SHRIRAMFIN.NS',
            'SHRIRAM': 'SHRIRAMFIN.NS',
        }
    
    def clean_ticker_input(self, ticker: str) -> str:
        """Clean and normalize ticker input"""
        ticker = str(ticker).upper().strip()
        # Remove common suffixes that might be added incorrectly
        ticker = re.sub(r'\.NS$|\.BO$', '', ticker)
        # Remove extra spaces and special characters
        ticker = re.sub(r'[^\w\s]', '', ticker)
        # Replace multiple spaces with single space
        ticker = re.sub(r'\s+', ' ', ticker)
        return ticker
    
    def get_proper_ticker_format(self, ticker: str) -> Tuple[str, str]:
        """
        Convert user input to proper ticker format with fallback options
        Returns (primary_symbol, fallback_symbol)
        """
        original_ticker = ticker
        ticker = self.clean_ticker_input(ticker)
        
        # Check exact matches first
        if ticker in self.symbol_mappings:
            return self.symbol_mappings[ticker], original_ticker
        
        # Try partial matches (first word or common patterns)
        words = ticker.split()
        if len(words) > 0:
            first_word = words[0]
            if first_word in self.symbol_mappings:
                return self.symbol_mappings[first_word], f"{first_word}.NS"
        
        # Check if already properly formatted
        if ticker.endswith('.NS') or ticker.endswith('.BO'):
            return ticker, ticker
        
        # Add .NS suffix as fallback
        fallback = f"{ticker}.NS"
        return fallback, original_ticker
    
    def validate_ticker_advanced(self, ticker: str) -> Tuple[bool, str, Dict]:
        """Advanced ticker validation with multiple fallback attempts"""
        try:
            # Get proper format
            primary_symbol, original = self.get_proper_ticker_format(ticker)
            
            # Test primary symbol
            test_data, used_symbol = self._test_symbol(primary_symbol)
            if not test_data.empty:
                company_info = self.get_company_info(used_symbol)
                return True, used_symbol, {
                    'original_input': original,
                    'used_symbol': used_symbol,
                    'company_name': company_info.get('name', 'N/A'),
                    'status': 'success'
                }
            
            # Try alternative approaches
            alternatives = self._generate_alternatives(ticker)
            for alt_symbol in alternatives:
                test_data, used_symbol = self._test_symbol(alt_symbol)
                if not test_data.empty:
                    company_info = self.get_company_info(used_symbol)
                    return True, used_symbol, {
                        'original_input': original,
                        'used_symbol': used_symbol,
                        'company_name': company_info.get('name', 'N/A'),
                        'status': 'alternative_success',
                        'tried_alternatives': True
                    }
            
            return False, primary_symbol, {
                'original_input': original,
                'used_symbol': primary_symbol,
                'status': 'not_found',
                'error': 'No valid symbol found'
            }
            
        except Exception as e:
            logger.error(f"Validation error for {ticker}: {str(e)}")
            return False, ticker, {'status': 'error', 'error': str(e)}
    
    def _test_symbol(self, symbol: str) -> Tuple[pd.DataFrame, str]:
        """Test if symbol exists and returns data"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period='5d', auto_adjust=True, prepost=False)
            return data, symbol
        except:
            return pd.DataFrame(), symbol
    
    def _generate_alternatives(self, ticker: str) -> list:
        """Generate alternative symbol formats to try"""
        ticker = self.clean_ticker_input(ticker)
        alternatives = []
        
        # Remove common words and try
        patterns = [
            r'^(.*?)(?:SERVICES?|INTERNATIONAL|INDIA|LIMITED|LTD)$',
            r'^(.*?)(?:BANK|FINANCE|FINANCIAL|CORP|CO)$',
            r'^(.*?)(?:GROUP|ENTERPRISES|PORTS|POWER|GAS)$'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, ticker, re.IGNORECASE)
            if match:
                base = match.group(1).strip()
                if base:
                    alternatives.append(f"{base}.NS")
                    alternatives.append(f"{base}.BO")
        
        # First word only
        words = ticker.split()
        if words:
            alternatives.append(f"{words[0]}.NS")
            alternatives.append(f"{words[0]}.BO")
        
        # Remove duplicates and original
        alternatives = list(set(alternatives) - {f"{ticker}.NS"})
        return alternatives[:5]  # Limit to 5 alternatives
    
    def get_stock_data(self, ticker: str, period: str = "1y", 
                      start_date: Optional[str] = None, 
                      end_date: Optional[str] = None,
                      validate: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """Fetch stock data with automatic symbol correction"""
        try:
            if validate:
                is_valid, used_symbol, info = self.validate_ticker_advanced(ticker)
                if not is_valid:
                    raise ValueError(f"Invalid ticker '{ticker}': {info.get('error', 'Symbol not found')}")
                ticker = used_symbol
                logger.info(f"Using symbol '{used_symbol}' for input '{info['original_input']}'")
            else:
                # Use basic format conversion
                ticker = self.get_proper_ticker_format(ticker)[0]
                info = {'used_symbol': ticker, 'original_input': ticker}
            
            cache_key = f"{ticker}_{period}_{start_date}_{end_date}"
            current_time = time.time()
            
            # Check cache
            if cache_key in self._cache:
                data, timestamp, cache_info = self._cache[cache_key]
                if current_time - timestamp < self.cache_duration:
                    logger.info(f"Returning cached data for {ticker}")
                    return data.copy(), cache_info
            
            stock = yf.Ticker(ticker)
            
            if start_date and end_date:
                data = stock.history(start=start_date, end=end_date, auto_adjust=True)
            else:
                data = stock.history(period=period, auto_adjust=True)
            
            if data.empty:
                raise ValueError(f"No data found for ticker: {ticker}")
            
            # Cache the data
            cache_info = {**info, 'data_points': len(data)}
            self._cache[cache_key] = (data.copy(), current_time, cache_info)
            
            logger.info(f"Successfully fetched {len(data)} data points for {ticker}")
            return data, cache_info
        
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame(), {'status': 'error', 'error': str(e)}
    
    def get_multiple_stocks(self, tickers: list, period: str = "1y") -> Dict[str, Dict]:
        """Fetch data for multiple stocks with validation"""
        results = {}
        for ticker in tickers:
            data, info = self.get_stock_data(ticker, period)
            results[ticker] = {'data': data, 'info': info}
        return results
    
    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive company information"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'name': info.get('longName', info.get('shortName', 'N/A')),
                'symbol': info.get('symbol', ticker),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'currency': info.get('currency', 'INR')
            }
        except Exception as e:
            logger.error(f"Error fetching company info for {ticker}: {str(e)}")
            return {'error': str(e)}
    
    def search_stocks(self, query: str, limit: int = 10) -> list:
        """Search for stocks by name or partial match"""
        query = self.clean_ticker_input(query)
        matches = []
        
        for key, symbol in self.symbol_mappings.items():
            if (query.lower() in key.lower() or 
                key.split()[0].lower() == query.lower()):
                try:
                    info = self.get_company_info(symbol)
                    matches.append({
                        'input_name': key,
                        'symbol': symbol,
                        'company_name': info.get('name', 'N/A'),
                        'sector': info.get('sector', 'N/A')
                    })
                    if len(matches) >= limit:
                        break
                except:
                    continue
        
        return matches