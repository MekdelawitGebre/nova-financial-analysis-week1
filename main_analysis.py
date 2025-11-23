import pandas as pd
# IMPORTANT: Ensure the FinancialAnalyzer class is imported correctly.
from financial_analyzer import FinancialAnalyzer
import os

# ==============================================================================
# --- CONFIGURATION (MUST BE RUN FIRST in a Notebook/Cell-based Environment) ---
# ==============================================================================
# Define the list of tickers to analyze
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA']
# Select one ticker to display detailed results and visualizations
example_ticker = 'AAPL' 
# Define the period for historical data download
START_DATE = '2023-01-01'
END_DATE = '2024-01-01'
# Path to your news data 
NEWS_FILE_PATH = 'data/news_data.csv'

# Ensure the data directory exists
os.makedirs('data', exist_ok=True)
os.makedirs('data/yfinanceData', exist_ok=True) 
# ==============================================================================


# --- 1. Initialization and Data Loading ---
print("--- 1. Initialization and Data Loading ---")
analyzer = FinancialAnalyzer(tickers)
# CRITICAL FIX: Setting force_download=True to ensure fresh, correctly named data
analyzer.load_stock_data(START_DATE, END_DATE, force_download=True)
analyzer.load_news_data(NEWS_FILE_PATH)


# --- 2. Descriptive Analysis (Returns & Volatility) ---
print("\n--- 2. Descriptive Analysis ---")
analyzer.perform_descriptive_analysis()


# --- 3. Time Series Analysis (Trends) ---
print("\n--- 3. Time Series Analysis ---")
analyzer.perform_time_series_analysis()


# --- 4. Technical Indicator Calculation ---
print("\n--- 4. Applying Technical Indicators (TA-Lib) ---")
for ticker in tickers:
    # This calls the method that uses the 'ta' library to calculate indicators
    analyzer.calculate_indicators(ticker)

# Display the calculated columns for verification
# NOTE: If running in cells, ensure you run the CONFIGURATION section above first 
# to define the 'example_ticker' variable and avoid a NameError.
if example_ticker in analyzer.stock_data and 'rsi' in analyzer.stock_data[example_ticker].columns:
    print(f"\nLast 5 rows of {example_ticker} with new Indicators:")
    # Displaying key indicator columns
    print(analyzer.stock_data[example_ticker].tail()[['close', 'rsi', 'macd', 'bb_mavg', 'bb_hband', 'bb_lband']])
else:
    print(f"\nCould not display indicator results for {example_ticker}. Check console for errors.")


# --- 5. Financial Metrics Calculation (Sharpe Ratio, MDD) ---
print("\n--- 5. Financial Metrics Calculation ---")
analyzer.calculate_financial_metrics()


# --- 6. Visualization ---
print("\n--- 6. Visualization ---")
analyzer.visualize_indicators(example_ticker)


# --- 7. NLP Analysis (Placeholder) ---
print("\n--- 7. NLP Analysis (Placeholder) ---")
analyzer.perform_nlp_analysis()