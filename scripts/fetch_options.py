#!/usr/bin/env python3
"""
Script: fetch_options.py

Description:
    This script uses the yfinance Python module to fetch options chain data
    for each NASDAQ-100 ticker. For each ticker, it downloads the available option
    expirations and fetches both calls and puts. The returned data—including strike,
    bid, ask, expiration, and other details—is stored in a SQLite database table
    called options_prices.

Requirements:
    - yfinance (install with: pip install yfinance)
    - pandas, sqlite3, logging, time
    - Proper network connectivity
"""

import os
import sqlite3
import datetime
import logging
import pandas as pd
from time import sleep
import yfinance as yf
from scripts.run_db_pipeline import get_nasdaq100_tickers  # Ensure this returns a list of ticker symbols
from scripts.config import DB_PATH, LOG_DIR, LOG_LEVEL

# Configure logging
log_file = os.path.join(LOG_DIR, "fetch_options.log")
logging.basicConfig(
    filename=log_file,
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Options Data Fetcher: Logging is active.")

def setup_database() -> sqlite3.Connection:
    """
    Create a database connection and ensure the options_prices table exists.
    This script drops any existing options_prices table and recreates it.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS options_prices")
        cursor.execute("""
            CREATE TABLE options_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                date TEXT,
                expiration_date TEXT,
                option_type TEXT,
                strike REAL,
                bid REAL,
                ask REAL,
                last_trade_price REAL,
                volume INTEGER,
                open_interest INTEGER,
                implied_volatility REAL,
                fetch_date TEXT
            )
        """)
        conn.commit()
        logging.info("Database table 'options_prices' created successfully.")
        return conn
    except Exception as e:
        logging.error(f"Database setup failed: {e}")
        raise

def fetch_options_chain_yfinance(ticker: str) -> pd.DataFrame:
    """
    Fetches the current options chain for a given ticker using yfinance.
    It iterates over all available option expirations, fetches both calls and puts,
    and merges them into a single DataFrame with standardized column names.
    
    Parameters:
        ticker (str): The ticker symbol.
        
    Returns:
        pd.DataFrame: A DataFrame with options data (strike, bid, ask, etc.)
    """
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            logging.info(f"No options expirations found for {ticker}.")
            return pd.DataFrame()
        data = pd.DataFrame()
        for exp_date in expirations:
            logging.info(f"Processing expiration {exp_date} for {ticker}.")
            try:
                options = tk.option_chain(exp_date)
                calls = options.calls
                puts = options.puts

                # Add option type column to each
                calls['option_type'] = 'C'
                puts['option_type'] = 'P'
                
                # Merge calls and puts and add expiration date column
                exp_data = pd.concat([calls, puts], ignore_index=True)
                exp_data['expiration_date'] = exp_date
                data = pd.concat([data, exp_data], ignore_index=True)
                
                # Delay between expiration fetches to be polite to Yahoo Finance
                sleep(0.5)
            except Exception as exp_err:
                logging.error(f"Error processing expiration {exp_date} for {ticker}: {exp_err}")
                continue

        if data.empty:
            return data

        # Add additional columns required by our database schema
        data['ticker'] = ticker
        current_date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        data['date'] = current_date_str
        data['fetch_date'] = current_date_str

        # Rename lastPrice to last_trade_price to match our database schema
        if 'lastPrice' in data.columns:
            data.rename(columns={'lastPrice': 'last_trade_price'}, inplace=True)

        # Reindex DataFrame to include only the required columns
        required_columns = [
            "ticker", "date", "expiration_date", "option_type", "strike", "bid", "ask",
            "last_trade_price", "volume", "open_interest", "implied_volatility", "fetch_date"
        ]
        data = data.reindex(columns=required_columns)
        return data
    except Exception as e:
        logging.error(f"Error fetching options chain for {ticker}: {e}")
        return pd.DataFrame()

def store_options_data(conn: sqlite3.Connection, tickers: list) -> None:
    """
    Iterates over each ticker, fetching the options chain data using yfinance,
    and storing it in the options_prices table.
    
    A delay is enforced between each ticker to avoid overloading Yahoo Finance.
    """
    cursor = conn.cursor()
    total_tickers = len(tickers)
    completed_tickers = 0
    for ticker in tickers:
        completed_tickers += 1
        progress = (completed_tickers / total_tickers) * 100
        progress_msg = f"Progress: {progress:.2f}% - Processing {ticker}"
        print(progress_msg)
        logging.info(progress_msg)
        try:
            print(f"Fetching options data for {ticker}")
            logging.info(f"Fetching options data for {ticker}")
            df = fetch_options_chain_yfinance(ticker)
            if df.empty:
                print(f"No options data for {ticker}")
                logging.info(f"No options data for {ticker}")
            else:
                cursor.executemany("""
                    INSERT INTO options_prices (ticker, date, expiration_date, option_type, strike, bid, ask,
                                                  last_trade_price, volume, open_interest, implied_volatility, fetch_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, df.values.tolist())
                conn.commit()
                print(f"Stored {len(df)} options for {ticker}")
                logging.info(f"Stored {len(df)} options for {ticker}")
        except Exception as e:
            error_msg = f"Error processing {ticker}: {e}"
            print(error_msg)
            logging.error(error_msg)
        # Delay between tickers to avoid hitting Yahoo Finance rate limits
        sleep(2)

def main():
    try:
        conn = setup_database()
        tickers = get_nasdaq100_tickers()
        store_options_data(conn, tickers)
        conn.close()
        print("Options data fetching and storage completed successfully.")
        logging.info("Options data fetching and storage completed successfully.")
    except Exception as e:
        print(f"Main process failed: {e}")
        logging.error(f"Main process failed: {e}")
        raise

if __name__ == "__main__":
    main()
