from tkinter import Image
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import warnings
import os 
import matplotlib.pyplot as plt

# --- Fix: Suppress specific FutureWarnings more aggressively ---
warnings.filterwarnings("ignore", category=FutureWarning, module='yfinance') 
warnings.simplefilter(action='ignore', category=FutureWarning) 
# ---------------------------------------------------------------------

class FinancialAnalyzer:
    """
    A class designed to perform comprehensive financial analysis, including
    loading stock data (with local persistence check), calculating technical 
    indicators, and analyzing related news and sentiment data.
    """
    def __init__(self, tickers: list):
        """
        Initializes the analyzer with a list of stock tickers.
        """
        self.tickers = tickers
        self.stock_data = {}
        self.news_data = None 

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper to ensure all relevant columns are lowercase and snake_case, 
        and explicitly verifies the presence of a primary closing price column ('close').
        """
        if df.empty:
            return df
        
        # 1. Standardize all column names to lowercase snake_case
        new_cols = {}
        for col in df.columns:
            # Convert to snake_case and lowercase
            standardized_col = str(col).strip().replace(' ', '_').replace('.', '_').lower()
            new_cols[col] = standardized_col
        
        df.rename(columns=new_cols, inplace=True)
        
        # 2. Ensure 'close' is the primary column name for the adjusted close price
        if 'adj_close' in df.columns and 'close' not in df.columns:
            # If yfinance returned 'Adj Close' but not 'Close', rename it.
            df.rename(columns={'adj_close': 'close'}, inplace=True)
        elif 'adj_close' in df.columns and 'close' in df.columns:
             # If both exist (rare with auto_adjust=True), drop the redundant 'adj_close' 
             # as the 'close' column is already the adjusted price.
             df.drop(columns=['adj_close'], inplace=True, errors='ignore')

        return df

    def load_stock_data(self, start_date: str, end_date: str, force_download=False) -> dict:
        """
        Loads historical stock data. Checks local CSV files first and saves them 
        locally if downloaded, ensuring column names are standardized.
        """
        DATA_FOLDER = 'data'
        YFINANCE_DATA_FOLDER = os.path.join(DATA_FOLDER, 'yfinanceData')
        
        os.makedirs(YFINANCE_DATA_FOLDER, exist_ok=True) 

        print(f"Loading stock data for {self.tickers}...")
        self.stock_data = {}
        
        for ticker_raw in self.tickers:
            ticker = ticker_raw.strip()
            if not ticker: continue
            
            file_path = os.path.join(YFINANCE_DATA_FOLDER, f'{ticker}.csv')
            df = pd.DataFrame() 

            # 1. Try Loading from Local CSV
            if os.path.exists(file_path) and not force_download:
                try:
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    # IMMEDIATELY standardize columns after loading from file
                    df = self._standardize_columns(df) 
                    print(f"Successfully loaded {ticker} from local file: {file_path}")
                except Exception as e:
                    print(f"Warning: Could not read local file {file_path}. Falling back to download. Error: {e}")
                    df = pd.DataFrame() 
            
            # 2. Download from YFinance if needed
            if df.empty or force_download:
                try:
                    # If forcing a download, delete the old file first to ensure cleanliness
                    if os.path.exists(file_path) and force_download:
                         os.remove(file_path)
                         print(f"Deleted old file for {ticker} to force fresh download.")
                         
                    print(f"Downloading {ticker} from Yahoo Finance (Period: {start_date} to {end_date})...")
                    # auto_adjust=True means only 'Open', 'High', 'Low', 'Close', 'Volume' are returned.
                    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                    
                    if df.empty:
                        print(f"Warning: No data found for ticker {ticker}.")
                        continue
                        
                except Exception as e:
                    print(f"Error loading data for {ticker}: {e}")
                    continue

            # --- Apply Standardization (Guarantees lowercase/snake_case) ---
            if not df.empty:
                df = self._standardize_columns(df)
                
                # Check for the critical 'close' column after standardization
                if 'close' not in df.columns:
                    print(f"Critical Error: 'close' column is missing for {ticker} even after standardization.")
                    continue
                
                # 3. Store and save
                self.stock_data[ticker] = df
                
                # Save the standardized data to CSV 
                if force_download or not os.path.exists(file_path):
                     df.to_csv(file_path)
                     print(f"Standardized and saved {ticker} to {file_path}")
                
        return self.stock_data


    def load_news_data(self, file_path: str):
        """Loads placeholder news data from a CSV file."""
        if not os.path.exists(file_path):
            print(f"Error: News data file not found at {file_path}")
            self.news_data = None
            return
            
        print(f"Loading news data from {file_path}...")
        try:
            self.news_data = pd.read_csv(file_path)
            required_cols = ['Source', 'Title', 'Rating']
            if not all(col in self.news_data.columns for col in required_cols):
                 print(f"Warning: News data missing one or more required columns ({required_cols}).")
            print("News data loading complete.")
        except Exception as e:
            print(f"Error loading news data: {e}")
            self.news_data = None


    def perform_descriptive_analysis(self):
        """Calculates descriptive statistics (mean, volatility, skewness, etc.) for daily returns."""
        if not self.stock_data:
            print("Error: Stock data is not loaded. Please call load_stock_data() first.")
            return {"results": "Stock data not available."}
            
        print("\n--- Performing Stock Data Descriptive Analysis (Returns & Volatility) ---")
        descriptive_results = {}

        for ticker, df in self.stock_data.items():
            # Check for standardized 'close' column
            if 'close' in df.columns and len(df) > 1:
                # Calculate Daily Returns (%)
                returns = df['close'].pct_change().dropna() * 100
                
                if returns.empty:
                    descriptive_results[ticker] = {"Error": "Insufficient data to calculate returns."}
                    continue

                # Calculate descriptive statistics
                stats = {
                    'mean_return_%': returns.mean().round(4),
                    'volatility_(std)': returns.std().round(4),
                    'skewness': returns.skew().round(4),
                    'kurtosis': returns.kurt().round(4),
                    'min_return_%': returns.min().round(4),
                    'max_return_%': returns.max().round(4)
                }
                descriptive_results[ticker] = stats
                print(f"\n[{ticker}] Daily Returns Stats:")
                for k, v in stats.items():
                    print(f"  {k}: {v}")
            else:
                descriptive_results[ticker] = {"Error": "Dataframe missing 'close' column or has too few rows."}
                print(f"Error for {ticker}: {descriptive_results[ticker]['Error']}")

        print("--- Stock Data Descriptive Analysis Complete ---")
        return descriptive_results


    def perform_time_series_analysis(self):
        """Performs basic time series analysis including SMA trend detection."""
        if not self.stock_data:
            print("Error: Stock data is not loaded. Please call load_stock_data() first.")
            return {"results": "Stock data not available."}

        print("\n--- Performing Stock Data Time Series Analysis (Trends & Volume) ---")
        time_series_results = {}
        
        short_window = 20
        long_window = 50

        for ticker, df in self.stock_data.items():
            
            if 'close' in df.columns and 'volume' in df.columns:
                
                # 1. Trend Detection using SMAs
                df['sma_short'] = df['close'].rolling(window=short_window).mean()
                df['sma_long'] = df['close'].rolling(window=long_window).mean()
                
                if len(df) < long_window:
                    time_series_results[ticker] = {"Error": f"Not enough data points for {long_window}-day SMA."}
                    print(f"Error for {ticker}: {time_series_results[ticker]['Error']}")
                    continue

                latest_close = df['close'].iloc[-1]
                latest_sma_short = df['sma_short'].iloc[-1]
                latest_sma_long = df['sma_long'].iloc[-1]

                trend_status = "Uptrend" if latest_close > latest_sma_short else "Downtrend"
                
                crossover = ""
                if latest_sma_short > latest_sma_long:
                    crossover = "Bullish Crossover (Short > Long)"
                else:
                    crossover = "Bearish Crossover (Short < Long)"

                # 2. Volume Analysis
                mean_volume = df['volume'].mean().round(0)
                
                # 3. Last 5-day % Change
                price_change_5d = ((latest_close / df['close'].iloc[-5]) - 1) * 100 if len(df) >= 5 else np.nan

                time_series_results[ticker] = {
                    'current_trend': trend_status,
                    'sma_crossover': crossover,
                    'latest_close': round(latest_close, 2),
                    'latest_20d_sma': round(latest_sma_short, 2),
                    'latest_50d_sma': round(latest_sma_long, 2),
                    'mean_daily_volume': int(mean_volume),
                    '5d_price_change_%': round(price_change_5d, 2) if not np.isnan(price_change_5d) else "N/A"
                }
                
                print(f"\n[{ticker}] Time Series Analysis:")
                for k, v in time_series_results[ticker].items():
                    print(f"  {k}: {v}")
            else:
                time_series_results[ticker] = {"Error": "Dataframe missing 'close' or 'volume' columns."}
                print(f"Error for {ticker}: {time_series_results[ticker]['Error']}")

        print("--- Stock Data Time Series Analysis Complete ---")
        return time_series_results


    def calculate_indicators(self, ticker: str):
        """
        Applies various technical analysis indicators (MACD, RSI, Bollinger Bands) 
        using TA-Lib.
        """
        if ticker in self.stock_data:
            df = self.stock_data[ticker]
            
            # Since load_stock_data now guarantees lowercase columns, we only need to check for existence
            if 'close' not in df.columns:
                print(f"Error: Required 'close' column not found in standardized data for {ticker}.")
                return
            
            print(f"Calculating technical indicators for {ticker}...")
            
            # All TA calculations now correctly reference the guaranteed lowercase 'close'
            
            # --- Relative Strength Index (RSI) ---
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            
            # --- Moving Average Convergence Divergence (MACD) ---
            macd_indicator = ta.trend.MACD(df['close'])
            df['macd'] = macd_indicator.macd()
            df['macd_signal'] = macd_indicator.macd_signal()
            df['macd_hist'] = macd_indicator.macd_diff() 
            
            # --- Bollinger Bands (BB) ---
            bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_hband'] = bb_indicator.bollinger_hband()
            df['bb_lband'] = bb_indicator.bollinger_lband()
            df['bb_mavg'] = bb_indicator.bollinger_mavg() 
            
            # Drop NaN values resulting from indicator calculations
            self.stock_data[ticker] = df.dropna().copy()
            print(f"Indicators calculated and NaNs dropped for {ticker}.")
        else:
            print(f"Error: Stock data for {ticker} not loaded.")


    def calculate_financial_metrics(self):
        """
        Calculates key portfolio financial metrics (KPIs) like Sharpe Ratio 
        and Maximum Drawdown for all loaded stocks.
        """
        if not self.stock_data:
            print("Error: Stock data is not loaded.")
            return {"results": "Stock data not available."}
        
        print("\n--- Calculating Financial Metrics (Sharpe Ratio, Max Drawdown) ---")
        metrics_results = {}
        TRADING_DAYS_PER_YEAR = 252
        RISK_FREE_RATE = 0.0 

        for ticker, df in self.stock_data.items():
            if 'close' in df.columns:
                # 1. Calculate Daily Returns
                df['daily_return'] = df['close'].pct_change()
                returns = df['daily_return'].dropna()
                
                if returns.empty:
                    metrics_results[ticker] = {"Error": "Insufficient data for returns calculation."}
                    continue

                # 2. Calculate Sharpe Ratio
                annualized_return = returns.mean() * TRADING_DAYS_PER_YEAR
                annualized_volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
                
                sharpe_ratio = (annualized_return - RISK_FREE_RATE) / annualized_volatility

                # 3. Calculate Maximum Drawdown (MDD)
                cumulative_returns = (1 + returns).cumprod()
                running_max = cumulative_returns.cummax()
                drawdown = (cumulative_returns / running_max) - 1
                max_drawdown = drawdown.min()
                
                metrics_results[ticker] = {
                    'annualized_return_%': round(annualized_return * 100, 2),
                    'annualized_volatility_%': round(annualized_volatility * 100, 2),
                    'sharpe_ratio': round(sharpe_ratio, 4),
                    'max_drawdown_%': round(max_drawdown * 100, 2)
                }
                
                print(f"\n[{ticker}] Financial Metrics:")
                for k, v in metrics_results[ticker].items():
                    print(f"  {k}: {v}")
            else:
                metrics_results[ticker] = {"Error": "Dataframe missing 'close' column."}

        print("--- Financial Metrics Calculation Complete ---")
        return metrics_results


    def visualize_indicators(self, ticker: str):
        """
        Creates a multi-panel visualization of the stock price, moving averages, 
        RSI, and MACD indicators.
        """
        if ticker not in self.stock_data or self.stock_data[ticker].empty:
            print(f"Error: Indicators for {ticker} are not calculated or data is empty.")
            return

        df = self.stock_data[ticker]
        
        if 'rsi' not in df.columns or 'macd' not in df.columns:
            print(f"Warning: Indicators (RSI/MACD) not found for {ticker}. Please call calculate_indicators() first.")
            return

        print(f"\n--- Generating Visualization for {ticker} ---")
        
        # Create 3 subplots: Price, RSI, MACD
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # 1. Price and Bollinger Bands Plot
        axes[0].plot(df['close'], label='Close Price', color='blue')
        axes[0].plot(df['bb_mavg'], label='BB Middle Band (20d SMA)', color='orange', linestyle='--')
        axes[0].plot(df['bb_hband'], label='BB Upper Band', color='red', linestyle=':')
        axes[0].plot(df['bb_lband'], label='BB Lower Band', color='green', linestyle=':')
        axes[0].set_title(f'{ticker} Price and Bollinger Bands')
        axes[0].legend(loc='upper left')
        axes[0].grid(True)
        
        # 2. RSI Plot
        axes[1].plot(df['rsi'], label='RSI (14)', color='purple')
        axes[1].axhline(70, color='red', linestyle='--', alpha=0.5)
        axes[1].axhline(30, color='green', linestyle='--', alpha=0.5)
        axes[1].set_title('Relative Strength Index (RSI)')
        axes[1].set_ylim(0, 100)
        axes[1].legend(loc='upper left')
        axes[1].grid(True)

        # 3. MACD Plot
        axes[2].plot(df['macd'], label='MACD', color='red')
        axes[2].plot(df['macd_signal'], label='Signal Line', color='blue')
        axes[2].bar(df.index, df['macd_hist'], label='Histogram', color=np.where(df['macd_hist'] > 0, 'green', 'red'), alpha=0.6)
        axes[2].set_title('MACD Indicator')
        axes[2].set_xlabel('Date')
        axes[2].legend(loc='upper left')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()  # Image of a stock technical analysis chart with price, RSI, and MACD indicators

        print(f"Visualization complete for {ticker}.")


    def perform_nlp_analysis(self):
        """
        Performs NLP and sentiment analysis on loaded news data.
        """
        if self.news_data is None:
            print("NLP analysis requires news data, which is currently not loaded.")
            return {"results": "NLP analysis placeholder."}
            
        print("Performing NLP and sentiment analysis...")
        
        if 'Sentiment_Score' in self.news_data.columns:
            avg_sentiment = self.news_data['Sentiment_Score'].mean()
            return {
                "average_sentiment_score": round(avg_sentiment, 4),
                "results": "Average sentiment calculated from placeholder data."
            }
        
        return {"results": "NLP analysis placeholder: Sentiment column not found."}