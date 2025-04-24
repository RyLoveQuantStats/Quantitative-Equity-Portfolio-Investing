#!/usr/bin/env python3
"""
Covered Call Strategy Based on Portfolio Holdings with Dynamic Rebalancing, Assignment Simulation,
Dynamic Volatility, and Transaction Costs.
=================================================================================
This script reads your portfolio positions stored in the 'optimized_hybrid_portfolios'
table and, for each analysis day, builds a covered call strategy for each long position.
For each ticker held it:
  1. Retrieves the underlying price.
  2. Determines an out‐of‐the‐money (OTM) strike price (set as a percentage above the price).
  3. Prices an American call using a binomial tree (CRR method) with early exercise.
  4. Estimates the number of shares held based on the current portfolio value.
  5. Retrieves the underlying price at expiration (analysis_date + expiry_days) and simulates
     assignment: if the price is above the strike, the call is assumed exercised (and shares are sold
     at the strike), otherwise the position is marked-to-market at the current price.
  6. Computes the total return per position (including premium collected minus transaction cost)
     and then rebalances the portfolio for the next period.
     
Per-ticker simulation details are stored in a new table ("covered_calls") and the daily updated
portfolio value is stored in ("portfolio_covered_calls"). In addition, the script calculates:
    - Original portfolio performance (from optimized_hybrid_portfolios)
    - Covered call (premium) performance from the simulated strategy, and
    - Combined portfolio performance – as the compounded return from the original strategy and the
      premium income.

Usage:
    python scripts/covered_calls.py --start 2020-01-01 --end 2024-12-31 --capital 100000000 --otm_delta 0.05 --expiry_days 30 --risk_free_rate 0.02 --volatility 0.25 --dividend_yield 0.02 --commission_rate 0.001 --vol_lookback 30 --use_dynamic_vol 1 --steps 200
"""

import sys
import os
import logging
import argparse
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta


import numpy as np
import pandas as pd
from math import log, sqrt, exp
from scipy.stats import norm
from typing import Tuple

# Path to the SQLite database file (assumed same as used for your portfolio)
DB_PATH = Path('database/data.db')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

##########################################################################
# BINOMIAL TREE OPTION PRICING FUNCTION (with adjustable steps)
##########################################################################
def binomial_tree_call_price(S: float, K: float, T: float, r: float, sigma: float,
                             q: float = 0.0, steps: int = 100) -> float:
    dt = T / steps
    u = exp(sigma * sqrt(dt))
    d = 1 / u
    p = (exp((r - q) * dt) - d) / (u - d)
    asset_prices = np.array([S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)])
    option_values = np.maximum(asset_prices - K, 0)
    for i in range(steps - 1, -1, -1):
        option_values = exp(-r * dt) * (p * option_values[1:i+2] + (1 - p) * option_values[0:i+1])
        asset_prices = np.array([S * (u ** j) * (d ** (i - j)) for j in range(i + 1)])
        option_values = np.maximum(option_values, asset_prices - K)
    return option_values[0]

##########################################################################
# EARLY EXERCISE RISK ESTIMATION FUNCTION
##########################################################################
def calculate_early_exercise_probability(S: float, K: float, option_price: float,
                                           dividend_yield: float, T: float) -> float:
    intrinsic = max(S - K, 0)
    time_value = option_price - intrinsic
    dividend_value = dividend_yield * S * T
    if time_value <= 0:
        return 1.0
    prob = dividend_value / time_value
    return max(0.0, min(prob, 1.0))

##########################################################################
# DYNAMIC VOLATILITY ESTIMATION (realized volatility over lookback period)
##########################################################################
def get_realized_volatility(conn: sqlite3.Connection, ticker: str, date: str, lookback_days: int = 30) -> float:
    end_date = datetime.strptime(date, "%Y-%m-%d")
    start_date = end_date - timedelta(days=lookback_days*2)
    query = """
        SELECT date, close FROM price
        WHERE ticker = ? AND date BETWEEN ? AND ?
        ORDER BY date
    """
    df = pd.read_sql_query(query, conn, params=(ticker, start_date.strftime("%Y-%m-%d"), date))
    if df.empty or len(df) < 10:
        return None
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    df['return'] = df['close'].pct_change()
    volatility = df['return'].std() * sqrt(252)
    return volatility

##########################################################################
# DATA ACCESS FUNCTIONS (for prices and portfolio positions)
##########################################################################
def get_underlying_price(conn: sqlite3.Connection, ticker: str, date: str) -> float:
    query = """
        SELECT close FROM price
        WHERE ticker = ? AND date = ?
        ORDER BY date LIMIT 1
    """
    df = pd.read_sql_query(query, conn, params=(ticker, date))
    if df.empty:
        logging.warning(f"No price data for ticker {ticker} on {date}.")
        return np.nan
    return float(df.iloc[0]['close'])

def get_future_price(conn: sqlite3.Connection, ticker: str, date: str, offset_days: int) -> float:
    start = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=offset_days - 2)
    end = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=offset_days + 2)
    query = """
        SELECT close FROM price
        WHERE ticker = ? AND date BETWEEN ? AND ?
        ORDER BY date LIMIT 1
    """
    df = pd.read_sql_query(query, conn, params=(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    if df.empty:
        return np.nan
    return float(df.iloc[0]['close'])

def read_portfolio_positions(conn: sqlite3.Connection, start_date: str, end_date: str) -> pd.DataFrame:
    query = """
        SELECT analysis_date, weights
        FROM optimized_hybrid_portfolios
        WHERE analysis_date BETWEEN ? AND ?
        ORDER BY analysis_date
    """
    df = pd.read_sql_query(query, conn, params=(start_date, end_date))
    records = []
    for idx, row in df.iterrows():
        analysis_date = row['analysis_date']
        try:
            weights = json.loads(row['weights'])
        except Exception as e:
            logging.error(f"Error parsing weights JSON for {analysis_date}: {e}")
            continue
        for ticker, weight in weights.items():
            records.append({'analysis_date': analysis_date, 'ticker': ticker, 'weight': weight})
    return pd.DataFrame(records)

##########################################################################
# COVERED CALL STRATEGY SIMULATION PER POSITION
##########################################################################
def simulate_covered_call_for_position(conn: sqlite3.Connection, S: float, otm_delta: float,
                                       expiry_days: int, r: float, sigma: float,
                                       dividend_yield: float, steps: int = 100) -> dict:
    K = S * (1 + otm_delta)
    T = expiry_days / 365.0
    call_price = binomial_tree_call_price(S, K, T, r, sigma, dividend_yield, steps=steps)
    early_ex_prob = calculate_early_exercise_probability(S, K, call_price, dividend_yield, T)
    return {
        'strike_price': K,
        'call_premium': call_price,
        'T_years': T,
        'early_ex_prob': early_ex_prob
    }

##########################################################################
# SIMULATE COVERED CALLS FOR ONE ANALYSIS DAY (INCLUDING ASSIGNMENT)
##########################################################################
def simulate_covered_calls_for_day(
    conn: sqlite3.Connection,
    analysis_date: str,
    current_capital: float,
    otm_delta: float,
    expiry_days: int,
    r: float,
    base_sigma: float,
    dividend_yield: float,
    commission_rate: float,
    use_dynamic_vol: bool,
    vol_lookback: int,
    steps: int = 100
) -> Tuple[pd.DataFrame, float]:
    """
    For a given analysis_date, this function:
      1. Retrieves portfolio positions.
      2. For each ticker, uses dynamic sigma if requested.
      3. Retrieves current price and prices the call via the binomial tree.
      4. Retrieves the underlying price at expiration (analysis_date + expiry_days).
      5. Determines assignment (if future price >= strike, sell at strike, else mark-to-market).
      6. Computes per-ticker total return = (final value + premium - initial cost - commissions).
    Returns a DataFrame with per-ticker simulation details and the updated portfolio value.
    """
    query = "SELECT weights FROM optimized_hybrid_portfolios WHERE analysis_date = ?"
    df = pd.read_sql_query(query, conn, params=(analysis_date,))
    if df.empty:
        logging.warning(f"No portfolio positions found for {analysis_date}.")
        return pd.DataFrame(), current_capital

    try:
        weights = json.loads(df.iloc[0]['weights'])
    except Exception as e:
        logging.error(f"Error parsing weights JSON for {analysis_date}: {e}")
        return pd.DataFrame(), current_capital

    results = []
    period_total = 0.0
    for ticker, weight in weights.items():
        S0 = get_underlying_price(conn, ticker, analysis_date)
        if np.isnan(S0):
            continue

        allocated_capital = current_capital * weight
        shares = allocated_capital / S0

        sigma = get_realized_volatility(conn, ticker, analysis_date, lookback_days=vol_lookback) if use_dynamic_vol else base_sigma
        if sigma is None:
            sigma = base_sigma

        option_data = simulate_covered_call_for_position(conn, S0, otm_delta, expiry_days, r, sigma, dividend_yield, steps)
        K = option_data['strike_price']
        premium = option_data['call_premium']

        future_date = (datetime.strptime(analysis_date, "%Y-%m-%d") + timedelta(days=expiry_days)).strftime("%Y-%m-%d")
        S_exp = get_future_price(conn, ticker, analysis_date, expiry_days)
        if np.isnan(S_exp):
            S_exp = S0

        final_price = K if S_exp >= K else S_exp

        cost_buy = commission_rate * allocated_capital
        cost_sell = commission_rate * (shares * final_price)

        position_return = (shares * final_price + shares * premium - cost_buy - cost_sell) - allocated_capital
        new_value = allocated_capital + position_return
        period_total += new_value

        results.append({
            'analysis_date': analysis_date,
            'ticker': ticker,
            'weight': weight,
            'initial_price': S0,
            'strike_price': K,
            'call_premium': premium,
            'shares': shares,
            'future_price': S_exp,
            'final_price': final_price,
            'allocated_capital': allocated_capital,
            'position_return': position_return,
            'new_value': new_value,
            'early_ex_prob': option_data['early_ex_prob']
        })

    new_portfolio_value = period_total
    day_df = pd.DataFrame(results)
    return day_df, new_portfolio_value

##########################################################################
# DATABASE STORAGE FUNCTIONS FOR COVERED CALL RESULTS & PORTFOLIO VALUE
##########################################################################
def ensure_covered_calls_table(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS covered_calls")  # Drop previous results
    cursor.execute("""
        CREATE TABLE covered_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            weight REAL,
            initial_price REAL,
            strike_price REAL,
            call_premium REAL,
            shares REAL,
            future_price REAL,
            final_price REAL,
            allocated_capital REAL,
            position_return REAL,
            new_value REAL,
            early_ex_prob REAL
        )
    """)
    conn.commit()
    logging.info("Ensured covered_calls table exists.")

def insert_covered_call_results(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO covered_calls (
                analysis_date, ticker, weight, initial_price, strike_price, call_premium,
                shares, future_price, final_price, allocated_capital, position_return,
                new_value, early_ex_prob
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row['analysis_date'],
            row['ticker'],
            row['weight'],
            row['initial_price'],
            row['strike_price'],
            row['call_premium'],
            row['shares'],
            row['future_price'],
            row['final_price'],
            row['allocated_capital'],
            row['position_return'],
            row['new_value'],
            row['early_ex_prob']
        ))
    conn.commit()
    logging.info("Inserted covered call results into the database.")

def ensure_portfolio_value_table(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS portfolio_covered_calls")
    cursor.execute("""
        CREATE TABLE portfolio_covered_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_date TEXT NOT NULL,
            portfolio_value REAL,
            period_return REAL
        )
    """)
    conn.commit()
    logging.info("Ensured portfolio_covered_calls table exists.")

def insert_portfolio_value(conn: sqlite3.Connection, analysis_date: str, portfolio_value: float, period_return: float) -> None:
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO portfolio_covered_calls (
            analysis_date, portfolio_value, period_return
        ) VALUES (?, ?, ?)
    """, (analysis_date, portfolio_value, period_return))
    conn.commit()

##########################################################################
# PERFORMANCE STATISTICS FUNCTIONS
##########################################################################
def calculate_covered_call_stats(conn: sqlite3.Connection, initial_capital: float,
                                 start_date: str, end_date: str, periods_per_year: int = 12) -> dict:
    query = """
        SELECT analysis_date, portfolio_value
        FROM portfolio_covered_calls
        WHERE analysis_date BETWEEN ? AND ?
        ORDER BY analysis_date
    """
    df = pd.read_sql_query(query, conn, params=(start_date, end_date))
    if df.empty:
        logging.error("No portfolio covered call values found for the specified date range.")
        return {}
    df.sort_values(by='analysis_date', inplace=True)
    df['return'] = df['portfolio_value'].pct_change().fillna(0)
    cumulative_return = (df['portfolio_value'].iloc[-1] / initial_capital) - 1
    arithmetic_mean = df['return'].mean()
    volatility = df['return'].std()
    annualized_return = (1 + cumulative_return) ** (periods_per_year / len(df)) - 1
    annualized_volatility = volatility * (periods_per_year ** 0.5)
    risk_free_rate = 0.02
    sharpe_ratio = ((annualized_return - risk_free_rate) / annualized_volatility) if annualized_volatility != 0 else np.nan
    stats = {
        'Covered Call Cumulative Return': cumulative_return,
        'Covered Call Arithmetic Mean Return': arithmetic_mean,
        'Covered Call Volatility': volatility,
        'Covered Call Annualized Return': annualized_return,
        'Covered Call Annualized Volatility': annualized_volatility,
        'Covered Call Sharpe Ratio': sharpe_ratio,
        'Number of Periods': len(df)
    }
    return stats

def calculate_overall_portfolio_stats(conn: sqlite3.Connection, start_date: str, end_date: str) -> dict:
    query = """
        SELECT analysis_date, portfolio_return
        FROM optimized_hybrid_portfolios
        WHERE analysis_date BETWEEN ? AND ?
        ORDER BY analysis_date
    """
    df = pd.read_sql_query(query, conn, params=(start_date, end_date))
    if df.empty:
        logging.error("No portfolio returns found for the specified period.")
        return {}
    df['analysis_date'] = pd.to_datetime(df['analysis_date'])
    df.sort_values(by='analysis_date', inplace=True)
    cumulative_return = (df['portfolio_return'] + 1).prod() - 1
    arithmetic_mean = df['portfolio_return'].mean()
    volatility = df['portfolio_return'].std()
    periods_per_year = 12
    annualized_return = (1 + cumulative_return) ** (periods_per_year / len(df)) - 1
    annualized_volatility = volatility * (periods_per_year ** 0.5)
    risk_free_rate = 0.02
    sharpe_ratio = ((annualized_return - risk_free_rate) / annualized_volatility) if annualized_volatility != 0 else np.nan
    stats = {
        'Portfolio Cumulative Return': cumulative_return,
        'Portfolio Arithmetic Mean Return': arithmetic_mean,
        'Portfolio Volatility': volatility,
        'Portfolio Annualized Return': annualized_return,
        'Portfolio Annualized Volatility': annualized_volatility,
        'Portfolio Sharpe Ratio': sharpe_ratio,
        'Number of Periods': len(df)
    }
    return stats

def calculate_combined_portfolio_stats(conn: sqlite3.Connection, initial_capital: float,
                                       start_date: str, end_date: str, periods_per_year: int = 12) -> dict:
    """
    Calculates the performance statistics for a combined strategy where for each analysis date
    the combined return is computed as:
         Combined_return = (1 + original_return) * (1 + premium_return) - 1.
    The function merges the returns from the original portfolio and the premium-based covered call simulation.
    """
    query_orig = """
        SELECT analysis_date, portfolio_return
        FROM optimized_hybrid_portfolios
        WHERE analysis_date BETWEEN ? AND ?
        ORDER BY analysis_date
    """
    query_cc = """
        SELECT analysis_date, period_return
        FROM portfolio_covered_calls
        WHERE analysis_date BETWEEN ? AND ?
        ORDER BY analysis_date
    """
    df_orig = pd.read_sql_query(query_orig, conn, params=(start_date, end_date))
    df_cc = pd.read_sql_query(query_cc, conn, params=(start_date, end_date))
    if df_orig.empty or df_cc.empty:
        logging.error("Insufficient data for combined portfolio statistics.")
        return {}
    # Ensure dates are datetime for merging
    df_orig['analysis_date'] = pd.to_datetime(df_orig['analysis_date'])
    df_cc['analysis_date'] = pd.to_datetime(df_cc['analysis_date'])
    # Merge on analysis_date (inner join)
    df_merge = pd.merge(df_orig, df_cc, on="analysis_date", how="inner")
    # Compute combined return for each period multiplicatively
    df_merge['combined_return'] = (1 + df_merge['portfolio_return']) * (1 + df_merge['period_return']) - 1
    cumulative_return = (1 + df_merge['combined_return']).prod() - 1
    arithmetic_mean = df_merge['combined_return'].mean()
    volatility = df_merge['combined_return'].std()
    annualized_return = (1 + cumulative_return) ** (periods_per_year / len(df_merge)) - 1
    annualized_volatility = volatility * (periods_per_year ** 0.5)
    risk_free_rate = 0.02
    sharpe_ratio = ((annualized_return - risk_free_rate) / annualized_volatility) if annualized_volatility != 0 else np.nan
    stats = {
        'Combined Cumulative Return': cumulative_return,
        'Combined Arithmetic Mean Return': arithmetic_mean,
        'Combined Volatility': volatility,
        'Combined Annualized Return': annualized_return,
        'Combined Annualized Volatility': annualized_volatility,
        'Combined Sharpe Ratio': sharpe_ratio,
        'Number of Periods': len(df_merge)
    }
    return stats

##########################################################################
# RUN COVERED CALL STRATEGY SIMULATION (Dynamic Portfolio Update)
##########################################################################
def run_portfolio_covered_call_simulation(conn: sqlite3.Connection, start_date: str, end_date: str,
                                          initial_capital: float, otm_delta: float, expiry_days: int,
                                          r: float, base_sigma: float, dividend_yield: float,
                                          commission_rate: float, use_dynamic_vol: bool, vol_lookback: int,
                                          steps: int = 100) -> None:
    """
    Iterates over each analysis_date, simulates the covered call for that day,
    updates the portfolio value using dynamic rebalancing and assignment simulation,
    and stores per-ticker results in covered_calls table and portfolio values in portfolio_covered_calls.
    """
    ensure_covered_calls_table(conn)
    ensure_portfolio_value_table(conn)

    query = "SELECT DISTINCT analysis_date FROM optimized_hybrid_portfolios WHERE analysis_date BETWEEN ? AND ? ORDER BY analysis_date"
    df_dates = pd.read_sql_query(query, conn, params=(start_date, end_date))
    if df_dates.empty:
        logging.error("No portfolio positions found in the specified date range.")
        return

    current_capital = initial_capital
    sorted_dates = df_dates['analysis_date'].tolist()
    for date in sorted_dates:
        logging.info(f"Simulating covered call strategy for {date} with starting capital {current_capital:.2f}...")
        day_df, new_capital = simulate_covered_calls_for_day(conn, date, current_capital,
                                                              otm_delta, expiry_days, r, base_sigma,
                                                              dividend_yield, commission_rate,
                                                              use_dynamic_vol, vol_lookback, steps)
        if not day_df.empty:
            insert_covered_call_results(conn, day_df)
        period_return = (new_capital - current_capital) / current_capital
        insert_portfolio_value(conn, date, new_capital, period_return)
        logging.info(f"Period return for {date}: {period_return:.4f}, new capital: {new_capital:.2f}")
        current_capital = new_capital

##########################################################################
# MAIN FUNCTION
##########################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Covered Call Strategy Based on Portfolio Holdings with Dynamic Rebalancing, "
                    "Assignment Simulation, Dynamic Volatility, and Transaction Costs."
    )
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument("--capital", type=float, default=100000000, help="Initial portfolio capital.")
    parser.add_argument("--otm_delta", type=float, default=0.05, help="OTM delta for strike price.")
    parser.add_argument("--expiry_days", type=int, default=30, help="Days until option expiry.")
    parser.add_argument("--risk_free_rate", type=float, default=0.02, help="Annual risk free rate.")
    parser.add_argument("--volatility", type=float, default=0.25, help="Base annualized volatility.")
    parser.add_argument("--dividend_yield", type=float, default=0.02, help="Annual dividend yield.")
    parser.add_argument("--commission_rate", type=float, default=0.001, help="Commission rate per transaction.")
    parser.add_argument("--vol_lookback", type=int, default=30, help="Lookback days for realized volatility.")
    parser.add_argument("--use_dynamic_vol", type=int, default=0, help="Flag to use dynamic volatility (1=yes).")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps in the binomial tree.")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    run_portfolio_covered_call_simulation(
        conn=conn,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        otm_delta=args.otm_delta,
        expiry_days=args.expiry_days,
        r=args.risk_free_rate,
        base_sigma=args.volatility,
        dividend_yield=args.dividend_yield,
        commission_rate=args.commission_rate,
        use_dynamic_vol=bool(args.use_dynamic_vol),
        vol_lookback=args.vol_lookback,
        steps=args.steps
    )
    conn.close()

if __name__ == "__main__":
    main()
