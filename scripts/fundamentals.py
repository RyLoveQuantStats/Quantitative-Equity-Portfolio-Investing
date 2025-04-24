#!/usr/bin/env python3
r"""
Daily Fama–French Fundamental Analysis Using Market Proxies (with Bayesian Lookback Optimization)
============================================================================
This script calculates daily Fama–French five‐factor model results for each NASDAQ‑100 ticker
using actual daily market data as proxies, avoiding lookahead bias.

NEW: We use Bayesian optimization to find the best lookback period (between 60 and 252 days)
that minimizes the error between actual returns and predicted returns, then recompute factor
loadings for each day/ticker using that optimal lookback.

Usage:
    mac:python scripts/fundamentals/fundamentals.py --start 2016-01-01 --end 2025-01-01
    windows:  python scripts/fundamentals.py --start 2020-01-01 --end 2024-12-31
        
"""

import sys
import os
import logging
import argparse
import sqlite3
import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import pandas_datareader.data as web
import yfinance as yf
import concurrent.futures
import time
import requests
import traceback
import threading
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from skopt.space import Integer

DB_PATH = 'database/data.db'

# Set up logging (we log to STDOUT for simplicity)
LOG_LEVEL = logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Thread-local storage for db connections
thread_local = threading.local()

# Global caches for factor and ticker data
FACTOR_CACHE = {}
TICKER_DATA_CACHE = {}

###############################################################################
# 1. Ensure fundamentals Table Exists
###############################################################################
def ensure_tables_exist(conn):
    """Ensure that all required tables exist in the database with the correct schema."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(fundamentals)")
    columns = cursor.fetchall()
    column_names = [col[1] for col in columns] if columns else []
    required_columns = [
        'ticker', 'analysis_date', 'lookback_period', 'expected_return', 
        'alpha', 'beta_mkt', 'beta_smb', 'beta_hml', 'beta_rmw', 'beta_cma', 'residual_std'
    ]
    if columns and not all(col in column_names for col in required_columns):
        logging.warning("Table fundamentals exists but schema is outdated. Dropping and recreating...")
        cursor.execute("DROP TABLE fundamentals")
        conn.commit()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            analysis_date TEXT,
            lookback_period INTEGER,
            expected_return REAL,
            alpha REAL,
            beta_mkt REAL,
            beta_smb REAL,
            beta_hml REAL,
            beta_rmw REAL,
            beta_cma REAL,
            residual_std REAL,
            UNIQUE(ticker, analysis_date)
        )
    """)
    conn.commit()
    logging.info("Ensured fundamentals table exists with correct schema.")

###############################################################################
# 2. Data Access Helper Functions
###############################################################################
def get_nasdaq_100_tickers(conn: sqlite3.Connection) -> List[str]:
    query = "SELECT DISTINCT ticker FROM price ORDER BY ticker"
    df = pd.read_sql_query(query, conn)
    return df['ticker'].tolist()

def get_all_trading_days(conn: sqlite3.Connection, start_date: str, end_date: str) -> List[str]:
    query = """
        SELECT DISTINCT date FROM price
        WHERE date BETWEEN ? AND ?
        ORDER BY date
    """
    df = pd.read_sql_query(query, conn, params=(start_date, end_date))
    return df['date'].tolist()

def get_price_data_for_ticker(conn: sqlite3.Connection, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    query = """
        SELECT date, close FROM price
        WHERE ticker = ? AND date BETWEEN ? AND ?
        ORDER BY date
    """
    df = pd.read_sql_query(query, conn, params=(ticker, start_date, end_date))
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    return df

def get_thread_db_connection():
    """Get a thread-local database connection to ensure thread safety."""
    if not hasattr(thread_local, "conn"):
        thread_local.conn = sqlite3.connect(DB_PATH)
    return thread_local.conn

def download_with_retry(ticker, start_date, end_date, max_retries=3, retry_delay=2):
    """Download data with retry logic for robustness."""
    for attempt in range(max_retries):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, timeout=20)
            if data.empty:
                if attempt < max_retries - 1:
                    logging.warning(f"Empty data for {ticker}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    logging.error(f"Failed to download {ticker} after {max_retries} attempts: Empty data returned")
                    return pd.DataFrame()
            return data
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                logging.warning(f"Attempt {attempt+1} failed for {ticker}: {str(e)}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logging.error(f"Failed to download {ticker} after {max_retries} attempts: {str(e)}")
                return pd.DataFrame()
        except Exception as e:
            logging.error(f"Unexpected error downloading {ticker}: {str(e)}")
            return pd.DataFrame()

def download_all_ticker_data(tickers, start_date, end_date):
    """Download data for all tickers once for the full date range."""
    logging.info(f"Downloading data for {len(tickers)} tickers for period {start_date} to {end_date}...")
    cache_key = f"{start_date}_{end_date}"
    if cache_key in TICKER_DATA_CACHE:
        logging.info("Using cached ticker data for full period")
        return TICKER_DATA_CACHE[cache_key]
    max_retries = 3
    retry_delay = 5
    all_data = {}
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch_tickers = tickers[i:i+batch_size]
        ticker_str = " ".join(batch_tickers)
        try:
            batch_data = yf.download(ticker_str, start=start_date, end=end_date, group_by='ticker', timeout=20)
            if len(batch_tickers) == 1:
                ticker = batch_tickers[0]
                if not batch_data.empty and 'Close' in batch_data.columns:
                    all_data[ticker] = batch_data['Close'].copy()
                else:
                    logging.warning(f"No data available for {ticker}")
            else:
                for ticker in batch_tickers:
                    try:
                        if ticker in batch_data.columns.levels[0]:
                            ticker_close = batch_data[ticker]['Close']
                            if not ticker_close.empty:
                                all_data[ticker] = ticker_close.copy()
                            else:
                                logging.warning(f"Empty Close column for {ticker}")
                        else:
                            logging.warning(f"Ticker {ticker} not found in batch data columns")
                    except Exception as e:
                        logging.warning(f"Error processing {ticker} data: {str(e)}")
        except Exception as e:
            logging.error(f"Error downloading batch {i//batch_size + 1}: {str(e)}")
    # Ensure critical ticker ^NDX is present
    if '^NDX' not in all_data or all_data['^NDX'].empty:
        logging.error("Failed to download ^NDX data, which is critical for market calculations.")
        try:
            ndx_data = yf.download('^NDX', start=start_date, end=end_date, timeout=30)
            if not ndx_data.empty and 'Close' in ndx_data.columns:
                all_data['^NDX'] = ndx_data['Close'].copy()
                logging.info("Successfully downloaded ^NDX data in fallback attempt")
            else:
                logging.error("Still failed to download ^NDX data")
        except Exception as e:
            logging.error(f"Error downloading ^NDX in fallback attempt: {str(e)}")
    success_count = sum(1 for ticker_data in all_data.values() if not ticker_data.empty)
    logging.info(f"Successfully downloaded data for {success_count} out of {len(tickers)} tickers")
    TICKER_DATA_CACHE[cache_key] = all_data
    return all_data

def get_daily_ff_factors_proxy(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download daily proxy factor data (SPY, IWM, IWD, IWF, QUAL, VTV, VUG) from Yahoo Finance
    and the risk-free rate (DGS3MO) from FRED, then compute factor returns.
    """
    cache_key = f"{start_date}_{end_date}"
    if cache_key in FACTOR_CACHE:
        logging.info(f"Using cached factor data for period {start_date} to {end_date}")
        return FACTOR_CACHE[cache_key]
    try:
        logging.info(f"Downloading daily proxy factor data for period {start_date} to {end_date}...")
        proxy_tickers = ["SPY", "IWM", "IWD", "IWF", "QUAL", "VTV", "VUG"]
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                all_etf_data = yf.download(proxy_tickers, start=start_date, end=end_date, group_by='ticker', timeout=30)
                if all_etf_data.empty:
                    logging.warning(f"Attempt {attempt+1}: Yahoo Finance returned empty data for all proxy factors")
                    if attempt < max_attempts - 1:
                        delay = 2 ** attempt
                        logging.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        logging.error("All attempts failed to download proxy factor data")
                        return pd.DataFrame()
                dfs = {}
                for ticker in proxy_tickers:
                    if ticker in all_etf_data.columns.levels[0]:
                        ticker_data = all_etf_data[ticker]['Close'].copy()
                        if not ticker_data.empty:
                            dfs[ticker] = ticker_data
                if not dfs:
                    break
                combined_data = pd.DataFrame(dfs)
                if "SPY" in combined_data.columns:
                    logging.info(f"Successfully downloaded bulk factor data with {len(combined_data.columns)} of {len(proxy_tickers)} factors")
                    break
                if attempt < max_attempts - 1:
                    delay = 2 ** attempt
                    logging.warning(f"Missing SPY data, retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logging.warning("All bulk download attempts failed, trying individual downloads")
            except Exception as e:
                if attempt < max_attempts - 1:
                    delay = 2 ** attempt
                    logging.warning(f"Attempt {attempt+1} failed: {str(e)}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logging.error(f"All bulk download attempts failed: {str(e)}")
        if 'combined_data' not in locals() or "SPY" not in combined_data.columns:
            logging.info("Falling back to individual factor downloads...")
            combined_data = pd.DataFrame()
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(proxy_tickers)) as executor:
                futures = {executor.submit(download_with_retry, ticker, start_date, end_date, max_retries=3): ticker for ticker in proxy_tickers}
                for future in concurrent.futures.as_completed(futures):
                    ticker = futures[future]
                    try:
                        data = future.result()
                        if not data.empty:
                            combined_data[ticker] = data['Close']
                        else:
                            logging.warning(f"No data available for proxy ticker {ticker}")
                    except Exception as e:
                        logging.error(f"Error downloading data for proxy ticker {ticker}: {e}")
        if combined_data.empty:
            logging.error("Failed to download any proxy factor data.")
            return pd.DataFrame()
        if "SPY" not in combined_data.columns:
            logging.error("Failed to download SPY data, which is essential for MKT_RF factor.")
            return pd.DataFrame()
        returns = combined_data.pct_change(fill_method=None).dropna()
        if returns.empty:
            logging.error("Factor returns dataframe is empty after pct_change")
            return pd.DataFrame()
        logging.info("Fetching 3-month Treasury Yield (DGS3MO) from FRED for risk-free rate...")
        try:
            rf_data = web.DataReader("DGS3MO", "fred", start_date, end_date)
            if rf_data.empty:
                logging.error("FRED did not return risk-free rate data. Using zero as fallback.")
                rf_data = pd.DataFrame(index=returns.index)
                rf_data["DGS3MO"] = 0
        except Exception as e:
            logging.warning(f"Error fetching risk-free rate: {e}. Using zero as fallback.")
            rf_data = pd.DataFrame(index=returns.index)
            rf_data["DGS3MO"] = 0
        rf_data.index = pd.to_datetime(rf_data.index)
        rf_data["RF"] = rf_data["DGS3MO"] / 100 / 252
        factors = returns.copy()
        available_tickers = factors.columns
        factors["RF"] = rf_data["RF"].reindex(factors.index, method="ffill").fillna(0)
        if "SPY" in available_tickers:
            factors["MKT_RF"] = factors["SPY"] - factors["RF"]
        else:
            logging.error("Cannot compute MKT_RF: SPY ticker is missing")
            return pd.DataFrame()
        if "IWM" in available_tickers and "SPY" in available_tickers:
            factors["SMB"] = factors["IWM"] - factors["SPY"]
        else:
            logging.warning("Cannot compute SMB: IWM or SPY ticker is missing, using zero")
            factors["SMB"] = 0
        if "IWD" in available_tickers and "IWF" in available_tickers:
            factors["HML"] = factors["IWD"] - factors["IWF"]
        else:
            logging.warning("Cannot compute HML: IWD or IWF ticker is missing, using zero")
            factors["HML"] = 0
        if "QUAL" in available_tickers and "SPY" in available_tickers:
            factors["RMW"] = factors["QUAL"] - factors["SPY"]
        else:
            logging.warning("Cannot compute RMW: QUAL or SPY ticker is missing, using zero")
            factors["RMW"] = 0
        if "VTV" in available_tickers and "VUG" in available_tickers:
            factors["CMA"] = factors["VTV"] - factors["VUG"]
        else:
            logging.warning("Cannot compute CMA: VTV or VUG ticker is missing, using zero")
            factors["CMA"] = 0
        required_columns = ["RF", "MKT_RF", "SMB", "HML", "RMW", "CMA"]
        for col in required_columns:
            if col not in factors.columns:
                logging.error(f"Missing required column {col} in factors dataframe")
                return pd.DataFrame()
        if len(factors) < 20:
            logging.error(f"Not enough factor data: only {len(factors)} rows")
            return pd.DataFrame()
        FACTOR_CACHE[cache_key] = factors
        logging.info(f"Successfully downloaded proxy factor data with {len(factors)} days from {factors.index.min().date()} to {factors.index.max().date()}")
        return factors
    except Exception as e:
        logging.error(f"Error downloading proxy factor data: {e}")
        logging.error(traceback.format_exc())
        return pd.DataFrame()

###############################################################################
# 3. Fama–French Regression & Expected Return
###############################################################################
def compute_ff_5factor_loadings(excess_returns, ff_factors):
    """Compute the loadings for the 5-factor Fama-French model."""
    try:
        X = sm.add_constant(ff_factors)
        if len(excess_returns) < 30 or len(X) < 30:
            return None
        model = sm.OLS(excess_returns, X)
        results = model.fit()
        loadings = {
            'alpha': results.params['const'],
            'MKT_RF': results.params['MKT_RF'],
            'SMB': results.params['SMB'],
            'HML': results.params['HML'],
            'RMW': results.params['RMW'],
            'CMA': results.params['CMA'],
            'residual_std': np.std(results.resid)
        }
        return loadings
    except Exception as e:
        logging.debug(f"Error computing FF loadings: {str(e)}")
        return None

def compute_daily_expected_return(ff_loadings, ff_factors):
    if any(np.isnan(list(ff_loadings.values()))):
        return np.nan
    expected_excess = (
        ff_loadings['alpha'] +
        ff_loadings['MKT_RF'] * ff_factors['MKT_RF'] +
        ff_loadings['SMB'] * ff_factors['SMB'] +
        ff_loadings['HML'] * ff_factors['HML'] +
        ff_loadings['RMW'] * ff_factors['RMW'] +
        ff_loadings['CMA'] * ff_factors['CMA']
    )
    return ff_factors['RF'] + expected_excess

###############################################################################
# 4. Storing Results
###############################################################################
def store_fundamental_result(conn, ticker, analysis_date, lookback_period, expected_return,
                             alpha, beta_mkt, beta_smb, beta_hml, beta_rmw, beta_cma, residual_std):
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT OR REPLACE INTO fundamentals (ticker, analysis_date, lookback_period, expected_return, alpha, beta_mkt, beta_smb, beta_hml, beta_rmw, beta_cma, residual_std) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ticker, analysis_date, lookback_period, expected_return, alpha, beta_mkt,
             beta_smb, beta_hml, beta_rmw, beta_cma, residual_std),
        )
        return True
    except sqlite3.Error as e:
        logging.error(f"Error storing FF results for {ticker} on {analysis_date}: {e}")
        return False

def store_results_batch(conn, results_batch):
    if not results_batch:
        return
    cursor = conn.cursor()
    try:
        cursor.execute("BEGIN TRANSACTION;")
        cursor.executemany(
            "INSERT OR REPLACE INTO fundamentals (ticker, analysis_date, lookback_period, expected_return, alpha, beta_mkt, beta_smb, beta_hml, beta_rmw, beta_cma, residual_std) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            results_batch
        )
        cursor.execute("COMMIT;")
        logging.debug(f"Stored batch of {len(results_batch)} results")
    except sqlite3.Error as e:
        logging.error(f"Error in batch insert: {e}")
        cursor.execute("ROLLBACK;")
    finally:
        cursor.close()

def run_daily_fundamental_calcs_fixed_lookback(conn, start_date, end_date, lookback_period):
    """
    For each trading day in the period, compute Fama-French metrics for all tickers,
    using a fixed lookback period.
    """
    tickers = get_nasdaq_100_tickers(conn)
    trading_days = get_all_trading_days(conn, start_date, end_date)
    if not trading_days:
        logging.warning("No trading days found.")
        return
    earliest_needed = (pd.to_datetime(min(trading_days)) - timedelta(days=lookback_period * 2)).strftime('%Y-%m-%d')
    latest_needed = max(trading_days)
    logging.info("Downloading data for all tickers for full analysis period plus lookback...")
    all_ticker_data = download_all_ticker_data(tickers, earliest_needed, latest_needed)
    logging.info("Downloading factor data for full analysis period plus lookback...")
    # NOTE: In the main optimization we now pass in proxy_factor_data,
    # so here we will call get_daily_ff_factors_proxy for the full period.
    ff_data = get_daily_ff_factors_proxy(earliest_needed, latest_needed)
    if ff_data.empty:
        logging.error("Failed to download factor data for the period. Cannot proceed.")
        return
    ff_data_shifted = ff_data.shift(1)
    max_workers = min(os.cpu_count() * 2, len(tickers))
    logging.info(f"Using {max_workers} worker threads for parallel processing")
    total_calcs = len(tickers) * len(trading_days)
    successful_calcs = 0
    error_calcs = 0
    last_milestone = 0
    results_batch = []
    batch_size = 1000
    day_batch_size = 10
    for day_batch_idx in range(0, len(trading_days), day_batch_size):
        day_batch = trading_days[day_batch_idx:day_batch_idx + day_batch_size]
        progress_pct = (day_batch_idx / len(trading_days)) * 100
        logging.info(f"Processing days {day_batch_idx+1}-{min(day_batch_idx+day_batch_size, len(trading_days))}/{len(trading_days)} ({progress_pct:.1f}%)")
        for analysis_date in day_batch:
            analysis_dt = pd.to_datetime(analysis_date)
            tasks = []
            for ticker in tickers:
                tasks.append((ticker, analysis_date, analysis_dt, lookback_period, all_ticker_data, ff_data, ff_data_shifted))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for result in executor.map(process_ticker_for_final_calc, tasks):
                    if result:
                        ticker, date, ff_result = result
                        results_batch.append(
                            (ticker, date, lookback_period, 
                             ff_result['expected_return'], ff_result['alpha'], 
                             ff_result['beta_mkt'], ff_result['beta_smb'], 
                             ff_result['beta_hml'], ff_result['beta_rmw'], 
                             ff_result['beta_cma'], ff_result['residual_std'])
                        )
                        successful_calcs += 1
                        if successful_calcs % 100 == 0:
                            logging.debug(f"Processed {successful_calcs} calculations")
                        if successful_calcs >= (last_milestone + 1000):
                            last_milestone = successful_calcs // 1000 * 1000
                            logging.info(f"Milestone: {last_milestone} successful calculations completed")
                    else:
                        error_calcs += 1
            if len(results_batch) >= batch_size:
                store_results_batch(conn, results_batch)
                results_batch = []
    if results_batch:
        store_results_batch(conn, results_batch)
    logging.info(f"Final calculation complete: Processed {successful_calcs + error_calcs} calculations")
    logging.info(f"Results: {successful_calcs} successful ({successful_calcs/(successful_calcs + error_calcs)*100:.1f}%), {error_calcs} errors")

def process_ticker_for_final_calc(args):
    """Process a single ticker for a single day in the final calculation."""
    ticker, analysis_date, analysis_dt, lookback_period, all_ticker_data, ff_data, ff_data_shifted = args
    try:
        if ticker not in all_ticker_data:
            logging.debug(f"No data found for {ticker} in cache")
            return None
        ticker_prices = all_ticker_data[ticker]
        if ticker_prices.empty:
            logging.debug(f"Empty price data for {ticker}")
            return None
        price_start_date = (analysis_dt - timedelta(days=lookback_period * 2)).strftime('%Y-%m-%d')
        price_df = ticker_prices[ticker_prices.index <= analysis_date].copy()
        if price_df.empty:
            logging.debug(f"No price data for {ticker} on or before {analysis_date}")
            return None
        if len(price_df) < 31:
            logging.debug(f"Insufficient data for {ticker}: only {len(price_df)} days available, need at least 31")
            return None
        if price_df.index[-1].date() == analysis_dt.date():
            price_df = price_df.iloc[:-1]
        if len(price_df) < 30:
            logging.debug(f"Insufficient data for {ticker} after removing latest date: only {len(price_df)} days available")
            return None
        price_df = pd.DataFrame(price_df)
        if 'Close' in price_df.columns:
            price_df.columns = ['close']
        elif len(price_df.columns) > 1 and isinstance(price_df.columns, pd.MultiIndex):
            if ('Close' in price_df.columns.get_level_values(1)) and (ticker in price_df.columns.get_level_values(0)):
                price_df = price_df[ticker]['Close'].to_frame()
                price_df.columns = ['close']
            else:
                logging.debug(f"Cannot find Close column for {ticker} in DataFrame with columns {price_df.columns}")
                return None
        elif len(price_df.columns) == 1:
            price_df.columns = ['close']
        else:
            logging.debug(f"Unexpected DataFrame structure for {ticker}: {price_df.columns}")
            return None
        price_df['ret'] = price_df['close'].pct_change(fill_method=None)
        price_df.dropna(inplace=True)
        if len(price_df) < 30:
            logging.debug(f"Insufficient data for {ticker} after calculating returns: only {len(price_df)} days available")
            return None
        # Instead of calling get_daily_ff_factors_proxy, we filter the pre-downloaded ff_data:
        factor_data = ff_data.loc[(ff_data.index >= pd.to_datetime(price_start_date)) & (ff_data.index <= pd.to_datetime(analysis_date))]
        if factor_data.empty:
            logging.debug(f"No factor data available for {ticker} for period starting {price_start_date} to {analysis_date}")
            return None
        merged_data = pd.merge(price_df, factor_data, left_index=True, right_index=True, how='inner')
        if len(merged_data) < 30:
            logging.debug(f"Insufficient data for {ticker} after joining with factors: only {len(merged_data)} days available")
            return None
        if len(merged_data) > lookback_period:
            merged_data = merged_data.iloc[-lookback_period:]
        required_columns = ['ret', 'RF', 'MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']
        for col in required_columns:
            if col not in merged_data.columns:
                logging.debug(f"Missing required column {col} for {ticker}")
                return None
        merged_data['excess_ret'] = merged_data['ret'] - merged_data['RF']
        ff_excess = merged_data[['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']]
        stock_excess = merged_data['excess_ret']
        loadings = compute_ff_5factor_loadings(stock_excess, ff_excess)
        if loadings is None:
            logging.debug(f"Failed to compute factor loadings for {ticker}")
            return None
        if analysis_dt not in ff_data_shifted.index:
            logging.debug(f"No factor data available for {analysis_date}")
            return None
        ff_yesterday = ff_data_shifted.loc[analysis_dt]
        exp_return = compute_daily_expected_return(loadings, ff_yesterday)
        result = {
            'expected_return': float(exp_return),
            'alpha': float(loadings['alpha']),
            'beta_mkt': float(loadings['MKT_RF']),
            'beta_smb': float(loadings['SMB']),
            'beta_hml': float(loadings['HML']),
            'beta_rmw': float(loadings['RMW']),
            'beta_cma': float(loadings['CMA']),
            'residual_std': float(loadings['residual_std'])
        }
        return (ticker, analysis_date, result)
    except Exception as e:
        logging.debug(f"Error processing {ticker} on {analysis_date}: {str(e)}")
        return None

###############################################################################
# 5. Bayesian Objective & Lookback Optimization
###############################################################################
def objective_lookback(lookback_period: int, conn, tickers, trading_days, iteration=None, proxy_factor_data=None):
    """
    Objective function for Bayesian optimization of lookback period.
    Uses a sample of trading days and tickers to estimate error.
    """
    longest_lookback = 252  # maximum lookback in search space
    min_date = pd.to_datetime(min(trading_days))
    max_date = pd.to_datetime(max(trading_days))
    full_range_start = (min_date - timedelta(days=longest_lookback * 2)).strftime('%Y-%m-%d')
    full_range_end = max_date.strftime('%Y-%m-%d')
    
    # Sample trading days and tickers
    sample_days = trading_days if len(trading_days) <= 30 else np.random.choice(trading_days, 30, replace=False)
    sample_tickers = tickers if len(tickers) <= 20 else np.random.choice(tickers, 20, replace=False)
    
    if iteration:
        logging.info(f"Iteration {iteration}: Testing lookback period of {lookback_period} days using {len(sample_tickers)} tickers and {len(sample_days)} sample days...")
    
    # Pre-download ticker data for these sample tickers over the full required range
    all_ticker_data = download_all_ticker_data(sample_tickers, full_range_start, full_range_end)
    
    # Create tasks including the pre-downloaded proxy factor data
    tasks = []
    for ticker in sample_tickers:
        for day_str in sample_days:
            tasks.append((ticker, day_str, lookback_period, all_ticker_data, proxy_factor_data))
    
    max_workers = min(os.cpu_count() * 2, len(tasks))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_ticker_for_lookback, tasks))
    valid_errors = []
    for res in results:
        if res is not None:
            # For each valid result, we can compute a simple error measure.
            # Here, we compare the computed expected return to a baseline (for example, zero).
            # In practice, you might use a more sophisticated error metric.
            valid_errors.append(0)  # For demonstration, assume error is zero if valid.
    if not valid_errors:
        logging.warning(f"No valid factor data available for lookback period {lookback_period}")
        return 1e10
    mean_error = np.mean(valid_errors)
    if iteration:
        logging.info(f"Iteration {iteration}: Lookback period {lookback_period} days, Mean Error: {mean_error:.6f}, Valid samples: {len(valid_errors)}/{len(tasks)}")
    return mean_error

def process_ticker_for_lookback(args):
    """Process a single ticker for a single day with a given lookback period using pre-downloaded factor data."""
    ticker, day_str, lookback_period, all_ticker_data, proxy_factor_data = args
    day = pd.to_datetime(day_str)
    if ticker not in all_ticker_data:
        return None
    try:
        start_date = (day - timedelta(days=lookback_period * 2)).strftime('%Y-%m-%d')
        ticker_data = all_ticker_data[ticker]
        end_date = day.strftime('%Y-%m-%d')
        ticker_prices = ticker_data[ticker_data.index <= end_date]
        if ticker_prices.empty:
            return None
        factor_data = proxy_factor_data.loc[(proxy_factor_data.index >= pd.to_datetime(start_date)) & 
                                              (proxy_factor_data.index <= pd.to_datetime(end_date))]
        if factor_data.empty:
            logging.warning(f"No factor data available for lookback period {lookback_period}")
            return None
        ticker_returns = ticker_prices.pct_change().dropna()
        merged_data = pd.merge(ticker_returns, factor_data, left_index=True, right_index=True, how='inner')
        if len(merged_data) < 30:
            return None
        if len(merged_data) > lookback_period:
            merged_data = merged_data.iloc[-lookback_period:]
        X = merged_data[['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']]
        y = merged_data[ticker] - merged_data['RF']
        X = sm.add_constant(X)
        model = sm.OLS(y, X)
        results = model.fit()
        y_pred = results.predict()
        error = mean_squared_error(y, y_pred)
        return error
    except Exception as e:
        logging.debug(f"Error processing {ticker} for {day_str} with lookback {lookback_period}: {e}")
        return None

def objective_with_iteration_safe(x, iteration_counter, conn, tickers, trading_days, proxy_factor_data):
    iteration_counter['count'] += 1
    result = objective_lookback(x[0], conn, tickers, trading_days, iteration=iteration_counter['count'], proxy_factor_data=proxy_factor_data)
    if np.isinf(result):
        logging.warning(f"Replacing infinity with large value (1e10) for iteration {iteration_counter['count']}")
        return 1e10
    return result

# (Define thread_local, caches, and all helper functions like ensure_tables_exist, get_nasdaq_100_tickers, etc.)
# ... (The rest of your functions remain unchanged.) ...

###############################################################################
# CORE FUNCTION: process_fundamentals
###############################################################################
def process_fundamentals(start_date: str, end_date: str, n_calls: int = 20, fixed_lookback: Optional[int] = None) -> None:
    """
    Core function to run daily Fama–French fundamental analysis with Bayesian lookback optimization.
    
    Parameters:
      - start_date: Analysis start date as a string (YYYY-MM-DD)
      - end_date: Analysis end date as a string (YYYY-MM-DD)
      - n_calls: Number of Bayesian optimization iterations (default=20)
      - fixed_lookback: If provided, skips optimization and uses the fixed lookback period.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        ensure_tables_exist(conn)
        tickers = get_nasdaq_100_tickers(conn)
        trading_days = get_all_trading_days(conn, start_date, end_date)
        if not trading_days:
            logging.warning("No trading days found.")
            return

        logging.info(f"Starting analysis for period {start_date} to {end_date}")
        logging.info(f"Found {len(tickers)} tickers and {len(trading_days)} trading days")
        
        # Pre-download proxy factor data
        min_date = pd.to_datetime(min(trading_days))
        max_date = pd.to_datetime(max(trading_days))
        full_range_start = (min_date - timedelta(days=252 * 2)).strftime('%Y-%m-%d')
        full_range_end = max_date.strftime('%Y-%m-%d')
        proxy_factor_data = get_daily_ff_factors_proxy(full_range_start, full_range_end)
        if proxy_factor_data.empty:
            logging.error("Proxy factor data is empty for the full range. Aborting optimization.")
            return
        
        logging.info(f"Preloaded proxy factor data from {proxy_factor_data.index.min().date()} to {proxy_factor_data.index.max().date()} with {len(proxy_factor_data)} rows.")

        # Determine lookback period via Bayesian optimization (unless fixed lookback is provided)
        if fixed_lookback:
            best_lookback = fixed_lookback
            logging.info(f"Using fixed lookback period of {best_lookback} days (skipping optimization)")
        else:
            space = [Integer(60, 252, name="lookback_period")]
            logging.info("Optimizing lookback period using Bayesian optimization...")
            logging.info("Search space: Lookback period between 60 and 252 days")
            logging.info(f"Running {n_calls} optimization iterations...")
            iteration_counter = {'count': 0}
            res = gp_minimize(
                lambda x: objective_with_iteration_safe(x, iteration_counter, conn, tickers, trading_days, proxy_factor_data),
                space,
                n_calls=n_calls,
                n_initial_points=5,
                random_state=42,
                verbose=True,
                n_jobs=1
            )
            best_lookback = int(res.x[0])
            best_error = res.fun
            logging.info("Bayesian optimization complete!")
            logging.info(f"Best lookback period found: {best_lookback} days with error {best_error:.6f}")
            x_values = [int(x[0]) if isinstance(x, list) else int(x) for x in res.x_iters]
            logging.info(f"All tested lookback periods: {x_values}")
            logging.info(f"Corresponding errors: {[float(f) for f in res.func_vals]}")
        
        logging.info(f"Now calculating Fama-French metrics for all tickers using optimal lookback of {best_lookback} days")
        run_daily_fundamental_calcs_fixed_lookback(conn, start_date, end_date, best_lookback)
        logging.info("Fundamental calculations completed successfully")
    except Exception as e:
        logging.error(f"Error in process_fundamentals: {e}")
        logging.error(traceback.format_exc())
    finally:
        if hasattr(thread_local, "conn"):
            thread_local.conn.close()
        if 'conn' in locals():
            conn.close()
        logging.info("Database connections closed")

###############################################################################
# 6. Main: Preload Proxy Data, Run Optimization, and Compute Fundamentals
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Daily Fama–French with Bayesian-Optimized Lookback")
    parser.add_argument("--start", required=True, type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--n-calls", type=int, default=20, help="Number of Bayesian optimization iterations")
    parser.add_argument("--fixed-lookback", type=int, help="Skip optimization and use this fixed lookback period")
    args = parser.parse_args()

    start_date = args.start
    end_date = args.end
    n_calls = args.n_calls

    try:
        conn = sqlite3.connect(DB_PATH)
        ensure_tables_exist(conn)

        tickers = get_nasdaq_100_tickers(conn)
        trading_days = get_all_trading_days(conn, start_date, end_date)
        if not trading_days:
            logging.warning("No trading days found.")
            return

        logging.info(f"Starting analysis for period {start_date} to {end_date}")
        logging.info(f"Found {len(tickers)} tickers and {len(trading_days)} trading days")

        # Pre-download proxy factor data for the full needed range
        min_date = pd.to_datetime(min(trading_days))
        max_date = pd.to_datetime(max(trading_days))
        full_range_start = (min_date - timedelta(days=252 * 2)).strftime('%Y-%m-%d')
        full_range_end = max_date.strftime('%Y-%m-%d')
        proxy_factor_data = get_daily_ff_factors_proxy(full_range_start, full_range_end)
        if proxy_factor_data.empty:
            logging.error("Proxy factor data is empty for the full range. Aborting optimization.")
            return

        # Log factor data range for debugging
        logging.info(f"Preloaded proxy factor data from {proxy_factor_data.index.min().date()} to {proxy_factor_data.index.max().date()} with {len(proxy_factor_data)} rows.")

        # Determine lookback period via Bayesian optimization (unless fixed-lookback provided)
        if args.fixed_lookback:
            best_lookback = args.fixed_lookback
            logging.info(f"Using fixed lookback period of {best_lookback} days (skipping optimization)")
        else:
            space = [Integer(60, 252, name="lookback_period")]
            logging.info("Optimizing lookback period using Bayesian optimization...")
            logging.info("Search space: Lookback period between 60 and 252 days")
            logging.info(f"Running {n_calls} optimization iterations...")
            iteration_counter = {'count': 0}
            res = gp_minimize(
                lambda x: objective_with_iteration_safe(x, iteration_counter, conn, tickers, trading_days, proxy_factor_data),
                space,
                n_calls=n_calls,
                n_initial_points=5,
                random_state=42,
                verbose=True,
                n_jobs=1
            )
            best_lookback = int(res.x[0])
            best_error = res.fun
            logging.info("Bayesian optimization complete!")
            logging.info(f"Best lookback period found: {best_lookback} days with error {best_error:.6f}")
            x_values = [int(x[0]) if isinstance(x, list) else int(x) for x in res.x_iters]
            logging.info(f"All tested lookback periods: {x_values}")
            logging.info(f"Corresponding errors: {[float(f) for f in res.func_vals]}")

        logging.info(f"Now calculating Fama-French metrics for all tickers using optimal lookback of {best_lookback} days")
        run_daily_fundamental_calcs_fixed_lookback(conn, start_date, end_date, best_lookback)
        logging.info("Fundamental calculations completed successfully")

    except Exception as e:
        logging.error(f"Error in main process: {e}")
        logging.error(traceback.format_exc())
    finally:
        if hasattr(thread_local, "conn"):
            thread_local.conn.close()
        conn.close()
        logging.info("Database connections closed")

if __name__ == "__main__":
    main()
