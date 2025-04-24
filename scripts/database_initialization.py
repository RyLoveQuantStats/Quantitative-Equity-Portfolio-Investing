#!/usr/bin/env python3
"""
init_database.py

This script initializes the database schema for our Integrated Quantitative Portfolio
Investment Strategy project. It creates the following tables with the specified columns:

1. Macro
   - CPI (REAL)
   - treasury_spread (REAL)  -- representing 10yr-2yr treasury spread

2. Price
   - ticker (TEXT)
   - date (TEXT)
   - open (REAL)
   - high (REAL)
   - low (REAL)
   - close (REAL)
   - volume (REAL)

3. Tickers
   - ticker (TEXT)  -- additional info can be added later

4. Fundamentals
   - ticker (TEXT)
   - analysis_date (TEXT)
   - lookback_period (INTEGER)
   - expected_return (REAL)
   - alpha (REAL)
   - beta_mkt (REAL)
   - beta_smb (REAL)
   - beta_hml (REAL)
   - beta_rmw (REAL)
   - beta_cma (REAL)
   - residual_std (REAL)
   -- Unique constraint on (ticker, analysis_date)

5. Hedging
   - date (TEXT)
   - ticker (TEXT)
   - beta (REAL)
   - nq_futures_hedge_amount (REAL)

6. Technicals (Technicals + DCA)
   - date (TEXT)
   - ticker (TEXT)
   - RSI_value (REAL)
   - signal (INTEGER)  -- Use 1 for TRUE, 0 for FALSE
   - amount_traded (REAL)

7. Backtest Full Strategy
   - date (TEXT)
   - sharpe_ratio (REAL)
   - win_rate (REAL)
   - total_return (REAL)
   - max_drawdown (REAL)
   - full_stats (TEXT)  -- JSON string or text of full backtest stats

Usage:
    python database_initialization.py
"""

import sys
import os
import sqlite3
import logging
from typing import List, Tuple
from scripts.config import DB_PATH, LOG_DIR, LOG_FILE, LOG_LEVEL
# Configure logging using centralized config values.
logging.basicConfig(
    filename=LOG_FILE,
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def drop_all_tables(conn: sqlite3.Connection):
    """Drop all tables in the connected SQLite database, skipping SQLite internal tables."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    for table in tables:
        table_name = table[0]
        # Skip internal SQLite tables
        if table_name.startswith("sqlite_"):
            continue
        print(f"Dropping table: {table_name}")
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    
    conn.commit()

def create_connection(db_file: str) -> sqlite3.Connection:
    """
    Create and return a database connection.
    """
    try:
        conn = sqlite3.connect(db_file)
        logging.info("Connected to database.")
        return conn
    except sqlite3.Error as e:
        logging.error(f"Database connection failed: {e}")
        raise

def execute_query(conn: sqlite3.Connection, query: str, params: Tuple = ()) -> None:
    """
    Execute a single query with optional parameters.
    """
    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        logging.info(f"Executed query: {query}")
    except sqlite3.Error as e:
        logging.error(f"Error executing query: {e}")
        raise

def create_table(conn: sqlite3.Connection, table_name: str, columns: List[Tuple[str, str]], table_constraints: str = "") -> None:
    """
    Create a table dynamically with specified columns.
    Each element in columns is a tuple: (column_name, column_type).
    Optionally, additional table constraints (e.g., UNIQUE constraints) can be provided as a string.
    """
    columns_str = ", ".join([f"{name} {col_type}" for name, col_type in columns])
    if table_constraints:
        columns_str = f"{columns_str}, {table_constraints}"
    query = f"CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY, {columns_str})"
    execute_query(conn, query)
    logging.info(f"Created table {table_name}")

if __name__ == "__main__":
    # Ensure the database directory exists if needed
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    conn = create_connection(DB_PATH)
    
    # Drop all existing tables (skipping internal SQLite tables)
    drop_all_tables(conn)

    # 1. Macro table
    create_table(
        conn,
        "macro",
        [("CPI", "REAL"), ("treasury_spread", "REAL")]
    )

    # 2. Price table
    create_table(
        conn,
        "price",
        [("ticker", "TEXT"), ("date", "TEXT"), ("open", "REAL"), ("high", "REAL"),
         ("low", "REAL"), ("close", "REAL"), ("volume", "REAL")]
    )

    # 3. Tickers table
    create_table(
        conn,
        "tickers",
        [("ticker", "TEXT")]
    )

    # 4. Fundamentals table with unique constraint on (ticker, analysis_date)
    create_table(
        conn,
        "fundamentals",
        [("ticker", "TEXT"),
         ("analysis_date", "TEXT"),
         ("lookback_period", "INTEGER"),
         ("expected_return", "REAL"),
         ("alpha", "REAL"),
         ("beta_mkt", "REAL"),
         ("beta_smb", "REAL"),
         ("beta_hml", "REAL"),
         ("beta_rmw", "REAL"),
         ("beta_cma", "REAL"),
         ("residual_std", "REAL")],
        table_constraints="UNIQUE(ticker, analysis_date)"
    )

    # 5. Hedging table
    create_table(
        conn,
        "hedging",
        [("date", "TEXT"),
         ("ticker", "TEXT"),
         ("beta", "REAL"),
         ("nq_futures_hedge_amount", "REAL")]
    )

    # 6. Technicals table (Technicals + DCA)
    create_table(
        conn,
        "technicals",
        [("date", "TEXT"),
         ("ticker", "TEXT"),
         ("RSI_value", "REAL"),
         ("signal", "INTEGER"),  # 1 for TRUE, 0 for FALSE
         ("amount_traded", "REAL")]
    )

    # 7. Backtest Full Strategy table
    create_table(
        conn,
        "backtest_full_strategy",
        [("date", "TEXT"),
         ("sharpe_ratio", "REAL"),
         ("win_rate", "REAL"),
         ("total_return", "REAL"),
         ("max_drawdown", "REAL"),
         ("full_stats", "TEXT")]
    )

    conn.close()
    logging.info("Database schema initialization complete.")
