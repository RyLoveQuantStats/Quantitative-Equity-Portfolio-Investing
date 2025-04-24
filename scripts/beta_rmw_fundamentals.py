#!/usr/bin/env python3
"""
Hybrid Portfolio Optimization Based on Beta_RMW & Monte Carlo Simulation
===========================================================================
This script implements a hybrid strategy which first filters stocks based on a 
smoothed beta_rmw signal (using a 21-day EMA) and only selects those stocks in the
top X% (e.g., top 10% when high_quantile=0.90). For these stocks it then retrieves 
their expected returns from fundamentals and historical price data to compute an 
annualized covariance matrix. A Monte Carlo simulation is run to generate a set of 
portfolios, and the portfolio maximizing the Sharpe ratio is chosen as the optimal
allocation for the holding period. The portfolio is held for a fixed period (e.g. 21 
trading days) before rebalancing.
  
Usage:
    python scripts/beta_rmw_fundamentals.py --output_dir ./results --high_quantile 0.60 --low_quantile 0.00 --rebalance_frequency 21 --roll_lookback 21
"""

import sys
import os
import logging
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import json
import time
from config import DB_PATH, RESULTS_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

##########################################################################
# DATA ACCESS FUNCTIONS
##########################################################################
def clear_output_table(conn: sqlite3.Connection) -> None:
    """
    Drop the 'optimized_hybrid_portfolios' table if it exists.
    """
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS optimized_hybrid_portfolios")
    conn.commit()
    logging.info("Cleared output table: optimized_hybrid_portfolios")

def get_trading_days(conn: sqlite3.Connection, start_date: str, end_date: str) -> List[str]:
    """Return a list of trading days from the price table between start_date and end_date."""
    query = """
        SELECT DISTINCT date FROM price
        WHERE date BETWEEN ? AND ?
        ORDER BY date
    """
    df = pd.read_sql_query(query, conn, params=(start_date, end_date))
    return df['date'].tolist()

def get_beta_rmw_for_date(conn: sqlite3.Connection, analysis_date: str) -> pd.Series:
    """Retrieve beta_rmw data for a given analysis_date from the fundamentals table."""
    query = """
        SELECT ticker, beta_rmw
        FROM fundamentals
        WHERE analysis_date = ?
    """
    df = pd.read_sql_query(query, conn, params=(analysis_date,))
    if df.empty:
        logging.warning(f"No beta_rmw data found for {analysis_date}")
        return pd.Series(dtype=float)
    return pd.Series(df['beta_rmw'].values, index=df['ticker'])

def get_expected_returns_for_date(conn: sqlite3.Connection, analysis_date: str) -> pd.Series:
    """Retrieve expected returns for a given analysis_date from the fundamentals table."""
    query = """
        SELECT ticker, expected_return
        FROM fundamentals
        WHERE analysis_date = ?
    """
    df = pd.read_sql_query(query, conn, params=(analysis_date,))
    if df.empty:
        logging.warning(f"No expected returns found for {analysis_date}")
        return pd.Series(dtype=float)
    return pd.Series(df['expected_return'].values, index=df['ticker'])

def get_price_data_for_tickers(conn: sqlite3.Connection, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieve close price data for a list of tickers between start_date and end_date.
    Returns a DataFrame with dates as index and tickers as columns.
    """
    placeholders = ",".join("?" for _ in tickers)
    query = f"""
        SELECT date, ticker, close
        FROM price
        WHERE ticker IN ({placeholders}) AND date BETWEEN ? AND ?
        ORDER BY date, ticker
    """
    params = tickers + [start_date, end_date]
    df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        logging.warning("No price data retrieved for tickers in the specified date range.")
        return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'])
    pivot_df = df.pivot(index='date', columns='ticker', values='close')
    return pivot_df.ffill().bfill()

def get_available_price_range(conn: sqlite3.Connection) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return the earliest and latest dates from the price table."""
    query = "SELECT MIN(date), MAX(date) FROM price"
    df = pd.read_sql_query(query, conn)
    if df.empty or pd.isna(df.iloc[0, 0]):
        logging.error("Price data is empty or missing.")
        return None, None
    return pd.to_datetime(df.iloc[0, 0]), pd.to_datetime(df.iloc[0, 1])

##########################################################################
# SIGNAL PROCESSING & WEIGHT CALCULATION FUNCTIONS
##########################################################################

def compute_rolling_beta_rmw(conn: sqlite3.Connection, tickers: List[str], end_date: str, roll_lookback: int) -> pd.Series:
    """
    Compute an exponential moving average (EMA) of beta_rmw values for a set of tickers 
    over the past 'roll_lookback' trading days up to end_date.
    """
    end_dt = pd.to_datetime(end_date)
    start_dt = end_dt - timedelta(days=roll_lookback * 2)
    query = f"""
        SELECT analysis_date, ticker, beta_rmw
        FROM fundamentals
        WHERE analysis_date BETWEEN ? AND ?
          AND ticker IN ({",".join("?" for _ in tickers)})
        ORDER BY analysis_date
    """
    params = [start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")] + tickers
    df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        logging.warning(f"No fundamentals data for EMA period ending {end_date}.")
        return pd.Series(dtype=float)
    df['analysis_date'] = pd.to_datetime(df['analysis_date'])
    pivot_df = df.pivot(index='analysis_date', columns='ticker', values='beta_rmw')
    ema = pivot_df.ewm(span=roll_lookback, adjust=False).mean()
    return ema.loc[:end_date].iloc[-1]

def calculate_beta_rmw_weights(beta_series: pd.Series, high_quantile: float, low_quantile: float) -> Tuple[Dict[str, float], float, float]:
    """
    Compute portfolio weights based on beta_rmw values.
    Only stocks with beta_rmw above the high_quantile (e.g. top 10% when high_quantile=0.90)
    are selected for long positions (equal-weighted). If low_quantile is 0.0, no shorts are taken.
    """
    if beta_series.empty:
        return {}, None, None
    high_thr = beta_series.quantile(high_quantile)
    low_thr = beta_series.quantile(low_quantile) if low_quantile > 0 else 0.0
    long_tickers = beta_series[beta_series >= high_thr].index.tolist()
    weights = {}
    if long_tickers:
        weight_long = 1.0 / len(long_tickers)
        for t in long_tickers:
            weights[t] = weight_long
    logging.info(f"EMA threshold: High >= {high_thr:.2f} (selected {len(long_tickers)} stocks)")
    return weights, high_thr, low_thr

##########################################################################
# MONTE CARLO SIMULATION FUNCTIONS
##########################################################################

def generate_random_weights(n_assets: int) -> np.ndarray:
    """Generate random weights that sum to 1 for n_assets."""
    weights = np.random.random(n_assets)
    return weights / np.sum(weights)

def calculate_portfolio_metrics(weights: np.ndarray, expected_returns: np.ndarray,
                                cov_matrix: np.ndarray, risk_free_rate: float = 0.02) -> Tuple[float, float, float]:
    """Compute portfolio expected return, volatility, and Sharpe ratio."""
    port_return = np.sum(weights * expected_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = ((port_return - risk_free_rate) / port_volatility) if port_volatility != 0 else np.nan
    return port_return, port_volatility, sharpe_ratio

def run_monte_carlo_simulation(expected_returns: pd.Series,
                               cov_matrix: pd.DataFrame,
                               analysis_date: str,
                               num_portfolios: int = 10000,
                               risk_free_rate: float = 0.02
                               ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run a Monte Carlo simulation for the given expected_returns and covariance matrix.
    Returns a DataFrame of simulated portfolio metrics and a dictionary with the portfolio
    that maximizes the Sharpe ratio.
    """
    n_assets = len(expected_returns)
    results = np.zeros((num_portfolios, 3 + n_assets))
    exp_ret_values = expected_returns.values
    cov_matrix_values = cov_matrix.values
    start_time = time.time()
    logging.info(f"Running Monte Carlo simulation with {num_portfolios} portfolios...")
    for i in range(num_portfolios):
        weights = generate_random_weights(n_assets)
        port_return, port_volatility, port_sharpe = calculate_portfolio_metrics(
            weights, exp_ret_values, cov_matrix_values, risk_free_rate
        )
        results[i, 0] = port_return
        results[i, 1] = port_volatility
        results[i, 2] = port_sharpe
        results[i, 3:] = weights
        if i % 1000 == 0 and i > 0:
            elapsed = time.time() - start_time
            logging.info(f"Completed {i} portfolios in {elapsed:.2f} seconds")
        if i % 5000 == 0 and i > 0:
            logging.debug(f"Monte Carlo simulation progress: reached {i} iterations")
    logging.info(f"Monte Carlo simulation completed in {time.time() - start_time:.2f} seconds")
    columns = ['return', 'volatility', 'sharpe_ratio'] + list(expected_returns.index)
    results_df = pd.DataFrame(results, columns=columns)
    # ── put this block back in ───────────────────────────────────────────
    opt_idx            = results_df['sharpe_ratio'].idxmax()
    optimal_portfolios = {'max_sharpe': results_df.loc[opt_idx]}
    # --------------------------------------------------------------------
    results_df['analysis_date'] = analysis_date
    results_df.to_csv(
        os.path.join(RESULTS_DIR, "mc_results_all.csv"),
        mode='a', header=not os.path.exists(os.path.join(RESULTS_DIR, "mc_results_all.csv")),
        index=False
    )
    return results_df, optimal_portfolios

##########################################################################
# PORTFOLIO RETURN & STORAGE FUNCTIONS
##########################################################################

def fetch_benchmark_returns(conn: sqlite3.Connection, start_date: str, end_date: str, ticker: str = "SPY") -> pd.Series:
    query = """
        SELECT date, close FROM price
        WHERE ticker = ? AND date BETWEEN ? AND ?
        ORDER BY date
    """
    df = pd.read_sql_query(query, conn, params=(ticker, start_date, end_date))
    if df.empty:
        logging.error(f"No benchmark data for {ticker} in the given date range.")
        return pd.Series()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    daily_ret = df['close'].pct_change().fillna(0)
    cum_ret = (1 + daily_ret).cumprod()
    return cum_ret

def get_hold_period_portfolio_return(conn: sqlite3.Connection, weights: Dict[str, float],
                                     start_date: str, end_date: str) -> float:
    """
    Compute the cumulative return over the holding period from start_date to end_date
    based on portfolio weights by calculating the return from the first to last available prices.
    """
    weighted_returns = []
    for ticker, weight in weights.items():
        query = """
            SELECT date, close
            FROM price
            WHERE ticker = ? AND date BETWEEN ? AND ?
            ORDER BY date
        """
        df = pd.read_sql_query(query, conn, params=(ticker, start_date, end_date))
        logging.debug(f"Ticker {ticker}: Retrieved {df.shape[0]} rows from {start_date} to {end_date}.")
        if not df.empty:
            logging.debug(f"Ticker {ticker} data snapshot:\n{df.head()}")
        if df.empty or len(df) < 2:
            logging.warning(f"Not enough data for {ticker} between {start_date} and {end_date}.")
            continue
        price_start = df.iloc[0]['close']
        price_end = df.iloc[-1]['close']
        try:
            ret = (price_end / price_start) - 1.0
        except Exception as e:
            logging.error(f"Error computing return for {ticker}: start={price_start}, end={price_end}, error={e}")
            continue
        weighted_ret = ret * weight
        logging.info(f"{ticker} holding period: start={price_start}, end={price_end}, raw_return={ret:.4f}, weight={weight:.4f}, weighted_return={weighted_ret:.4f}")
        weighted_returns.append(weighted_ret)
    if not weighted_returns:
        logging.error(f"No valid returns computed for portfolio from {start_date} to {end_date}.")
    return sum(weighted_returns) if weighted_returns else np.nan

def ensure_output_table(conn: sqlite3.Connection) -> None:
    """
    Ensure the output table 'optimized_hybrid_portfolios' exists with the proper schema.
    """
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(optimized_hybrid_portfolios)")
    columns = cursor.fetchall()
    required = {'analysis_date', 'portfolio_return', 'expected_return', 'volatility', 'sharpe_ratio', 'weights'}
    if columns:
        existing = {col[1] for col in columns}
        if not required.issubset(existing):
            logging.warning("Table schema outdated; dropping and recreating table.")
            cursor.execute("DROP TABLE optimized_hybrid_portfolios")
            conn.commit()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS optimized_hybrid_portfolios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_date TEXT NOT NULL,
            portfolio_return REAL,
            expected_return REAL,
            volatility REAL,
            sharpe_ratio REAL,
            weights TEXT
        )
    """)
    conn.commit()
    logging.info("Output table (optimized_hybrid_portfolios) ensured.")

def store_portfolio_result(conn: sqlite3.Connection, analysis_date: str,
                           portfolio_metrics: Dict[str, float], weights: Dict[str, float], period_return: float) -> None:
    """
    Store the hybrid portfolio result into the database.
    """
    cursor = conn.cursor()
    try:
        weights_json = json.dumps(weights)
        cursor.execute("""
            INSERT INTO optimized_hybrid_portfolios (
                analysis_date, portfolio_return, expected_return, volatility, sharpe_ratio, weights
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (analysis_date,
              period_return,
              portfolio_metrics.get('return', np.nan),
              portfolio_metrics.get('volatility', np.nan),
              portfolio_metrics.get('sharpe_ratio', np.nan),
              weights_json))
        conn.commit()
        logging.info(f"Stored hybrid portfolio for {analysis_date}: Return = {period_return:.2%}")
    except Exception as e:
        logging.error(f"Error storing portfolio for {analysis_date}: {e}")

##########################################################################
# HYBRID BACKTESTING ENGINE
##########################################################################

def run_hybrid_strategy(conn: sqlite3.Connection, start_date: str, end_date: str,
                        high_quantile: float, low_quantile: float,
                        rebalance_frequency: int, roll_lookback: int, price_lookback: int,
                        num_portfolios: int, risk_free_rate: float, output_dir: str) -> None:
    """
    Runs the hybrid strategy that first screens stocks based on EMA–smoothed beta_rmw (top X%),
    then for those stocks retrieves expected returns and price history, and runs a Monte Carlo
    simulation to optimize for a high Sharpe ratio portfolio for the holding period.
    """
    all_days = get_trading_days(conn, start_date, end_date)
    logging.info(f"Found {len(all_days)} trading days between {start_date} and {end_date}")
    if not all_days:
        logging.error("No trading days found in the specified period.")
        return

    ensure_output_table(conn)
    hybrid_returns = {}

    # Rebalancing dates
    # Generate monthly rebalancing dates: select the first trading day of each month.
    pd_all_days = pd.to_datetime(all_days)
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    monthly_dates = pd.date_range(start=start_ts, end=end_ts, freq='MS')
    rebalance_dates = []
    for d in monthly_dates:
        valid_days = pd_all_days[pd_all_days >= d]
        if not valid_days.empty:
            rebalance_dates.append(str(valid_days[0].date()))
    # Ensure that the last trading day is included.
    if all_days[-1] not in rebalance_dates:
        rebalance_dates.append(all_days[-1])

    for i in range(len(rebalance_dates) - 1):
        current_date = rebalance_dates[i]
        next_date = rebalance_dates[i+1]
        logging.info(f"Rebalancing on {current_date}; holding until {next_date}")

        # Step 1: Retrieve and smooth beta_rmw via EMA
        beta_series = get_beta_rmw_for_date(conn, current_date)
        tickers = beta_series.index.tolist()
        if roll_lookback > 1:
            beta_series = compute_rolling_beta_rmw(conn, tickers, current_date, roll_lookback)
        if beta_series.empty:
            logging.warning(f"No beta_rmw signal on {current_date}; skipping period.")
            continue

        # Step 2: Screen stocks with beta_rmw above high_quantile threshold (long-only)
        screening_weights, high_thr, _ = calculate_beta_rmw_weights(beta_series, high_quantile, low_quantile)
        if not screening_weights:
            logging.warning(f"No stocks qualify on {current_date}; skipping period.")
            continue

        # Step 3: Retrieve expected returns for current date and keep only screened stocks
        exp_returns = get_expected_returns_for_date(conn, current_date)
        screened_tickers = list(screening_weights.keys())
        exp_returns = exp_returns.loc[exp_returns.index.intersection(screened_tickers)]
        logging.debug(f"Expected returns retrieved for {len(exp_returns)} tickers on {current_date}: {exp_returns.index.tolist()}")
        if exp_returns.empty:
            logging.warning(f"No expected returns for screened stocks on {current_date}; skipping period.")
            continue

        # Step 4: Retrieve price data for the selected stocks over a lookback period for covariance
        lookback_start = (pd.to_datetime(current_date) - timedelta(days=price_lookback * 2)).strftime("%Y-%m-%d")
        lookback_end = (pd.to_datetime(current_date) - timedelta(days=1)).strftime("%Y-%m-%d")
        price_data = get_price_data_for_tickers(conn, list(exp_returns.index), lookback_start, lookback_end)
        if price_data.empty or price_data.shape[0] < price_lookback:
            logging.warning(f"Insufficient price data from {lookback_start} to {lookback_end} on {current_date}")
            continue

        returns_df = price_data.pct_change().dropna()
        cov_matrix = returns_df.cov() * 252  # Annualized covariance
        logging.info(f"Covariance matrix computed with shape {cov_matrix.shape}. Min value: {cov_matrix.min().min():.6f}, Max value: {cov_matrix.max().max():.6f}")
        # Step 5: Run Monte Carlo simulation on the selected stocks
        try:
            sim_results, opt_portfolios = run_monte_carlo_simulation(
                exp_returns, cov_matrix, current_date,
                num_portfolios=num_portfolios, risk_free_rate=risk_free_rate
            )
        except Exception as e:
            logging.error(f"Monte Carlo simulation error on {current_date}: {e}")
            continue

        optimal_port = opt_portfolios.get('max_sharpe')
        if optimal_port is None:
            logging.warning(f"No optimal portfolio found on {current_date}; skipping period.")
            continue

        # The simulation output: columns 'return', 'volatility', 'sharpe_ratio' followed by asset weights.
        # Extract the Monte Carlo–optimal weights.
        mc_weights = optimal_port.iloc[3:].to_dict()

        # Optionally, you might blend mc_weights with the screening weights.
        # Here we use the Monte Carlo–derived weights as our allocation.
        optimized_weights = mc_weights

        # Step 6: Compute realized return during the holding period using the optimal weights.
        period_return = get_hold_period_portfolio_return(conn, optimized_weights, current_date, next_date)
        logging.debug(f"Computed holding period return for {current_date} to {next_date}: {period_return:.4f}")

        if np.isnan(period_return):
            logging.warning(f"Failed to compute return for period {current_date} to {next_date}.")
            continue

        # For record keeping, store simulation metrics:
        portfolio_metrics = {
            'return': optimal_port['return'],
            'volatility': optimal_port['volatility'],
            'sharpe_ratio': optimal_port['sharpe_ratio']
        }
        store_portfolio_result(conn, current_date, portfolio_metrics, optimized_weights, period_return)
        hybrid_returns[current_date] = period_return
        logging.info(f"Period {current_date} to {next_date}: Return = {period_return:.2%}")

    # Save cumulative returns
    if hybrid_returns:
        ret_series = pd.Series(hybrid_returns)
        ret_series.index = pd.to_datetime(ret_series.index)
        cumulative_returns = (1 + ret_series.sort_index()).cumprod()
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        file_path = out_path / f"hybrid_portfolio_returns_{start_date}_to_{end_date}.csv"
        cumulative_returns.to_csv(file_path, header=["Cumulative Return"])
        logging.info(f"Cumulative returns saved to {file_path}")
    else:
        logging.error("No portfolio returns recorded in the hybrid strategy.")

##########################################################################
# PORTFOLIO STATISTICS FUNCTION
##########################################################################

def calculate_overall_portfolio_stats(conn: sqlite3.Connection, start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Calculate overall portfolio performance based on stored hybrid portfolio returns.
    Returns a dictionary of key metrics (cumulative return, annualized return, volatility, Sharpe ratio, etc.).
    """
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
    period_returns = df['portfolio_return']

    cumulative_return = (1 + period_returns).prod() - 1
    arithmetic_mean = period_returns.mean()
    geometric_mean = (1 + period_returns).prod() ** (1 / len(period_returns)) - 1
    volatility = period_returns.std()
    periods_per_year = 12  # adjust if necessary
    annualized_return = (1 + cumulative_return) ** (periods_per_year / len(period_returns)) - 1
    annualized_volatility = volatility * (periods_per_year ** 0.5)
    risk_free_rate = 0.02
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else np.nan

    cum_returns = (1 + period_returns).cumprod()
    max_drawdown = (cum_returns - cum_returns.cummax()).min()

    stats = {
        'Cumulative Return': cumulative_return,
        'Arithmetic Mean Return': arithmetic_mean,
        'Geometric Mean Return': geometric_mean,
        'Volatility (per period)': volatility,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Maximum Drawdown': max_drawdown,
        'Number of Periods': len(period_returns)
    }
    return stats

##########################################################################
# MAIN FUNCTION
##########################################################################

def main() -> None:
    np.random.seed(42)
    parser = argparse.ArgumentParser(
        description="Hybrid Beta_RMW & Monte Carlo Optimization: Maximize Sharpe Ratio & Expected Return."
    )
    parser.add_argument("--start", type=str,
                        help="Backtest start date (YYYY-MM-DD). If not provided, use earliest from fundamentals.")
    parser.add_argument("--end", type=str,
                        help="Backtest end date (YYYY-MM-DD). If not provided, use latest from fundamentals.")
    parser.add_argument("--high_quantile", default=0.90, type=float,
                        help="Quantile threshold for beta_rmw screening (default=0.90; top 10% stocks)")
    parser.add_argument("--low_quantile", default=0.00, type=float,
                        help="Quantile threshold for shorts (default=0.00; no shorts)")
    parser.add_argument("--rebalance_frequency", default=21, type=int,
                        help="Rebalancing frequency in trading days (default=21)")
    parser.add_argument("--roll_lookback", default=21, type=int,
                        help="EMA lookback span for beta_rmw (default=21)")
    parser.add_argument("--price_lookback", default=126, type=int,
                        help="Lookback period (in trading days) for price data to compute covariance (default=126)")
    parser.add_argument("--num_portfolios", default=10000, type=int,
                        help="Number of Monte Carlo portfolios (default=10000)")
    parser.add_argument("--risk_free_rate", default=0.02, type=float,
                        help="Annual risk free rate (default=0.02)")
    parser.add_argument("--output_dir", default=".", type=str,
                        help="Directory for output CSV files (default=current directory)")
    args = parser.parse_args()

    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            # Clear previous results
            clear_output_table(conn)
            
            # Now ensure the output table exists with the proper schema
            ensure_output_table(conn)
            
            # Set default start/end if not provided based on fundamentals date range.
            if not args.start or not args.end:
                query = "SELECT MIN(analysis_date), MAX(analysis_date) FROM fundamentals"
                df_dates = pd.read_sql_query(query, conn)
                if df_dates.empty or pd.isna(df_dates.iloc[0, 0]):
                    logging.error("No fundamentals date range found.")
                    sys.exit(1)
                if not args.start:
                    args.start = df_dates.iloc[0, 0]
                    logging.info(f"Using earliest fundamentals date: {args.start}")
                if not args.end:
                    args.end = df_dates.iloc[0, 1]
                    logging.info(f"Using latest fundamentals date: {args.end}")
            # Ensure price history is sufficient.
            earliest_price, _ = get_available_price_range(conn)
            if earliest_price is None:
                logging.error("Price data unavailable.")
                sys.exit(1)
            if pd.to_datetime(args.start) < earliest_price:
                args.start = earliest_price.strftime("%Y-%m-%d")
                logging.info(f"Adjusted start date to {args.start} due to price data limits.")

            logging.info(f"Running hybrid strategy from {args.start} to {args.end}")
            run_hybrid_strategy(
                conn,
                start_date=args.start,
                end_date=args.end,
                high_quantile=args.high_quantile,
                low_quantile=args.low_quantile,
                rebalance_frequency=args.rebalance_frequency,
                roll_lookback=args.roll_lookback,
                price_lookback=args.price_lookback,
                num_portfolios=args.num_portfolios,
                risk_free_rate=args.risk_free_rate,
                output_dir=args.output_dir
            )

            stats = calculate_overall_portfolio_stats(conn, args.start, args.end)
            if stats:
                logging.info("Overall Hybrid Portfolio Statistics:")
                for key, value in stats.items():
                    logging.info(f"  {key}: {value}")
    except sqlite3.Error as e:
        logging.error(f"SQLite error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.exception("Unexpected error during hybrid backtest.")
        sys.exit(1)

if __name__ == "__main__":
    main()
