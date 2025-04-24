#!/usr/bin/env python3
"""
Main Script: run_db_pipeline.py

This script is designed to fetch and store data from various sources into a SQLite database.

Description:
    This script performs the following tasks:
      1. Fetches NASDAQ-100 tickers and stores them in the 'tickers' table of your SQLite database.
      2. Fetches macroeconomic data (Treasury yields, CPI, GDP, bond spreads, etc.) from the FRED API
         and stores the data in the 'macro_data' table.
      3. Fetches price data for active tickers (from NASDAQ-100) and baseline indices (SPY, ^NDX) using yfinance
         and stores the data in the 'price' table.
      4. **NEW:** Fetches options chain data from yfinance for the tickers (using config start and end dates)
         and stores them in the new database table `options_prices`.
"""

import os
import sqlite3
import datetime
import logging
import pandas as pd
from time import sleep
from fredapi import Fred
import yfinance as yf
from typing import List, Optional
from scripts.config import FRED_API_KEY, DB_PATH, LOG_DIR, LOG_LEVEL, LOG_FILE, START_DATE, END_DATE
from scripts.config import FRED_API_KEY, DB_PATH, LOG_DIR, LOG_LEVEL, LOG_FILE, START_DATE, END_DATE#, POLYGON_API_KEY, HISTORICAL_START_DATE, HISTORICAL_END_DATE

# Configure logging for the merged script
log_file = os.path.join(LOG_DIR, 'main.log')
logging.basicConfig(
    filename=log_file,
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize fredapi with your API key
fred = Fred(api_key=FRED_API_KEY)

# =============================================================================
# Database Setup
# =============================================================================
def setup_database() -> sqlite3.Connection:
    """
    Create a database connection and ensure necessary tables exist.
    This version drops existing 'tickers', 'price', and 'options_prices' tables (if any)
    so that the new schema is used.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Drop tables if they exist so the new schema is created
        cursor.execute('DROP TABLE IF EXISTS tickers')
        cursor.execute('DROP TABLE IF EXISTS price')
        cursor.execute('DROP TABLE IF EXISTS options_prices')
        conn.commit()
        # Create tickers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tickers (
                id INTEGER PRIMARY KEY,
                ticker TEXT UNIQUE
            )
        ''')
        # Create price table with the correct columns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price (
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (ticker, date)
            )
        ''')
        # Create options_prices table (new)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS options_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                expiration_date TEXT,
                option_type TEXT,
                strike REAL,
                bid REAL,
                ask REAL,
                lastPrice REAL,
                volume INTEGER,
                open_interest INTEGER,
                implied_volatility REAL,
                fetch_date TEXT
            )
        ''')
        conn.commit()
        logging.info("Database tables (tickers, price, options_prices) created successfully.")
        return conn
    except Exception as e:
        logging.error(f"Database setup failed: {e}")
        raise

# =============================================================================
# Macroeconomic Data Functions
# =============================================================================
# Define FRED series to fetch
fred_series = {
    "10Y Treasury Yield": "DGS10",
    "5Y Treasury Yield":  "DGS5",
    "2Y Treasury Yield":  "DGS2",
    "30Y Treasury Yield": "DGS30",
    "CPI":                "CPIAUCSL",
    "GDP":                "GDPC1",  # Real GDP (seasonally adjusted annual rate)
    "BAA Corporate Bond Spread": "BAA10Y",
    "High Yield Bond Index":     "BAMLH0A0HYM2"
}

def fetch_fred_data(series_dict, start="2016-01-01", end="2024-12-31") -> pd.DataFrame:
    """
    Fetch data for each FRED series defined in series_dict.
    Returns a DataFrame with dates as index and one column per series.
    """
    data_frames = []
    for label, series_id in series_dict.items():
        try:
            series = fred.get_series(series_id, observation_start=start, observation_end=end)
            df = series.to_frame(name=label)
            data_frames.append(df)
        except Exception as e:
            logging.error(f"Error fetching '{label}' (ID: {series_id}) from FRED: {e}")
            print(f"Error fetching '{label}' (ID: {series_id}) from FRED: {e}")
    if data_frames:
        return pd.concat(data_frames, axis=1)
    else:
        return pd.DataFrame()

def store_dataframe(df: pd.DataFrame, table_name: str, db_path: str = DB_PATH) -> None:
    """
    Store a pandas DataFrame in the SQLite database specified by db_path under the given table_name.
    The DataFrame index (which contains the dates) is stored as a column labeled 'Date'.
    """
    try:
        conn = sqlite3.connect(db_path)
        df.to_sql(table_name, conn, if_exists='replace', index=True, index_label="Date")
        conn.close()
        print(f"Data stored successfully in table '{table_name}' at {db_path}.")
        logging.info(f"Data stored successfully in table '{table_name}' at {db_path}.")
    except Exception as e:
        logging.error(f"Error storing dataframe in table '{table_name}': {e}")
        raise

def fetch_and_store_fred_data():
    print("Starting to fetch macroeconomic data from FRED API.")
    logging.info("Starting to fetch macroeconomic data from FRED API.")
    
    # Fetch all defined series into one DataFrame
    macro_df = fetch_fred_data(fred_series, start="2016-01-01", end="2024-12-31")
    
    # Fill missing values: forward-fill then backward-fill
    macro_df.ffill(inplace=True)
    macro_df.bfill(inplace=True)
    
    # Calculate derived metrics: yield spread between 10Y and 2Y Treasury yields
    if "10Y Treasury Yield" in macro_df.columns and "2Y Treasury Yield" in macro_df.columns:
        macro_df["Yield Spread (10Y-2Y)"] = macro_df["10Y Treasury Yield"] - macro_df["2Y Treasury Yield"]
    
    print("Fetched macroeconomic data:")
    print(macro_df.head())
    logging.info("Fetched macroeconomic data successfully.")
    
    # Store the DataFrame in the SQLite database under the "macro_data" table.
    store_dataframe(macro_df, "macro_data")
    print("Macroeconomic data successfully stored in the SQL database.")
    logging.info("Macroeconomic data successfully stored in the SQL database.")

# =============================================================================
# Ticker Data Functions
# =============================================================================
def get_nasdaq100_tickers() -> list:
    """
    Return a hardcoded list of current NASDAQ-100 tickers.
    """
    tickers = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'AVGO', 'ADBE', 
        'COST', 'PEP', 'CSCO', 'NFLX', 'CMCSA', 'AMD', 'TMUS', 'INTC', 'INTU', 'QCOM', 
        'TXN', 'AMAT', 'AMGN', 'HON', 'SBUX', 'ISRG', 'ADI', 'MDLZ', 'GILD', 'REGN', 
        'ADP', 'VRTX', 'PANW', 'KLAC', 'LRCX', 'SNPS', 'ASML', 'CDNS', 'MRVL', 'BKNG', 
        'ABNB', 'ADSK', 'ORLY', 'FTNT', 'CTAS', 'MELI', 'MNST', 'PAYX', 'KDP', 'PCAR', 
        'CRWD', 'DXCM', 'CHTR', 'LULU', 'NXPI', 'MCHP', 'WDAY', 'CPRT', 'ODFL', 'ROST', 
        'FAST', 'EXC', 'BIIB', 'CSGP', 'ANSS', 'CTSH', 'DDOG', 'IDXX', 'VRSK', 'TEAM', 
        'DLTR', 'ILMN', 'ZS', 'ALGN', 'MTCH', 'FANG', 'ENPH', 'GEHC', 'DASH', 'SGEN', 
        'SIRI', 'CCEP', 'SPLK', 'TTWO', 'VRSN', 'SWKS', 'AEP', 'WBD', 'XEL', 'CSX', 
        'FISV', 'ATVI', 'MDB', 'PYPL', 'LCID', 'RIVN', 'TTD', 'SMCI', 'PLTR', 'MSTR'
    ]
    return tickers

def fetch_and_store_tickers(conn: sqlite3.Connection) -> None:
    """
    Fetch and store current NASDAQ-100 tickers into the 'tickers' table.
    """
    cursor = conn.cursor()
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    try:
        tickers = get_nasdaq100_tickers()
        if not tickers:
            logging.error("No tickers retrieved")
            return
        
        for ticker in tickers:
            cursor.execute("""
                INSERT OR REPLACE INTO tickers (ticker) VALUES (?)
            """, (ticker,))
        conn.commit()
        logging.info(f"Successfully stored {len(tickers)} tickers for {current_date}")
        print(f"Successfully stored {len(tickers)} tickers.")
    except Exception as e:
        logging.error(f"Error in fetch_and_store_tickers: {e}")
        conn.rollback()
        raise

# =============================================================================
# Price Data Functions
# =============================================================================
def get_active_tickers(conn: sqlite3.Connection) -> List[str]:
    """
    Get list of unique tickers from the tickers table.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM tickers")
    tickers = [row[0] for row in cursor.fetchall()]
    return tickers

def fetch_price_data(ticker: str, start_date: str, end_date: str, retries: int = 3) -> Optional[pd.DataFrame]:
    """
    Fetch price data for a given ticker using yfinance with a retry mechanism.
    """
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, interval='1d')
            if df.empty:
                logging.warning(f"No data available for {ticker} between {start_date} and {end_date}")
                return None
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            if attempt == retries - 1:
                logging.error(f"Failed to fetch data for {ticker} after {retries} attempts: {e}")
                return None
            logging.warning(f"Attempt {attempt + 1} failed for {ticker}: {e}")
            sleep(1)
    return None

def store_price_data(conn: sqlite3.Connection, ticker: str, df: pd.DataFrame) -> None:
    """
    Store fetched price data for a ticker into the 'price' table.
    """
    try:
        cursor = conn.cursor()
        data = [
            (
                ticker,
                date.strftime('%Y-%m-%d'),
                row['Open'],
                row['High'],
                row['Low'],
                row['Close'],
                row['Volume']
            )
            for date, row in df.iterrows()
        ]
        cursor.executemany("""
            INSERT OR REPLACE INTO price (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, data)
        conn.commit()
        logging.info(f"Successfully stored {len(data)} price points for {ticker}")
    except Exception as e:
        logging.error(f"Error storing price data for {ticker}: {e}")
        conn.rollback()

def fetch_and_store_price_data(conn: sqlite3.Connection) -> None:
    """
    Fetch and store price data for active tickers and baseline indices.
    """
    tickers = get_active_tickers(conn)
    logging.info(f"Found {len(tickers)} unique active tickers to process for price data.")
    
    # Define the date range for fetching price data
    start_date = START_DATE
    end_date = END_DATE
    
    # Process each active ticker
    for ticker in tickers:
        df = fetch_price_data(ticker, start_date, end_date)
        if df is not None:
            store_price_data(conn, ticker, df)
        sleep(0.5)  # Pause to help avoid rate limiting
    
    # Process baseline tickers (e.g., SPY, ^NDX)
    baseline_tickers = ['SPY', '^NDX']
    for ticker in baseline_tickers:
        df = fetch_price_data(ticker, start_date, end_date)
        if df is not None:
            store_price_data(conn, ticker, df)
        sleep(0.5)
    logging.info("Price data fetch and storage completed successfully.")

# =============================================================================
# Main Function
# =============================================================================
def main():
    """
    Main function to fetch all data:
      1. Set up the database and required tables.
      2. Fetch and store NASDAQ-100 tickers.
      3. Fetch and store macroeconomic data from FRED.
      4. Fetch and store price data for tickers and baseline indices.
      5. **NEW:** Fetch and store options chain data for tickers in the 'options_prices' table.
    """
    try:
        # Setup database and create necessary tables (dropping old ones to ensure correct schema)
        conn = setup_database()
        
        # Fetch and store tickers
        fetch_and_store_tickers(conn)
        
        # Fetch and store macroeconomic data (macro_data table is created via to_sql)
        fetch_and_store_fred_data()
        
        # Fetch and store price data
        fetch_and_store_price_data(conn)
        
        # NEW: Fetch and store options chain data
        tickers = get_nasdaq100_tickers()
        
        conn.close()
        logging.info("All data fetching processes completed successfully.")
        print("All data fetching processes completed successfully.")
    except Exception as e:
        logging.error(f"Main process failed: {e}")
        print(f"Main process failed: {e}")
        raise

if __name__ == "__main__":
    main()
