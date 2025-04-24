#!/usr/bin/env python3
"""
Combined Strategy: Hybrid Portfolio + Covered Calls, Benchmark Comparison,
and Risk Hedging via VaR/CVaR
=======================================================================
This script performs three layers of logic:
  1.  **Hybrid equity portfolio** – stocks are screened with a Beta‑RMW signal
      and fed into a Monte‑Carlo (MC) optimiser to maximise Sharpe‐ratio.
  2.  **Covered‑call overlay** – a binomial‑tree engine writes calls against
      the equity allocation and tracks cash‑flows/premiums.
  3.  **Analytics** – merges the equity & option legs, compares them with
      SPY, and computes risk metrics (VaR / CVaR + simple hedge ratio).

Run example
-----------
    python -m scripts.combined_strategy_rmw_calls \
    --start 2020-01-01 --end 2024-12-31 \
    --capital 100000000 --otm_delta 0.05 --expiry_days 45 \
    --risk_free_rate 0.02 --volatility 0.25 --dividend_yield 0.02 \
    --commission_rate 0.001 --vol_lookback 30 --use_dynamic_vol \
    --steps 200 --equity_weight 0.7
"""

# ──────────────────────────────────────────────────────────────────────
#  Imports & constants
# ──────────────────────────────────────────────────────────────────────
import argparse
import json
import logging
import os
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3

from scripts.covered_calls import (
    calculate_covered_call_stats,
    calculate_overall_portfolio_stats as cc_portfolio_stats,  # renamed to avoid clash
    ensure_covered_calls_table,
    ensure_portfolio_value_table,
    insert_covered_call_results,
    insert_portfolio_value,
    run_portfolio_covered_call_simulation as covered_call_sim,
    simulate_covered_calls_for_day,
)
from scripts.beta_rmw_fundamentals import (
    calculate_beta_rmw_weights,
    calculate_overall_portfolio_stats,
    clear_output_table,
    compute_rolling_beta_rmw,
    fetch_benchmark_returns,
    get_beta_rmw_for_date,
    get_expected_returns_for_date,
    get_price_data_for_tickers,
    get_trading_days,
    run_hybrid_strategy as hybrid_engine,
    run_monte_carlo_simulation,
)
from scripts.config import DB_PATH, RESULTS_DIR

# matplotlib one‑liner for nicer DPI when saving files
plt.rcParams["figure.dpi"] = 110

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)5s  | %(message)s",
)

# ──────────────────────────────────────────────────────────────────────
#  Helper functions (unchanged business logic, only cleaned imports)
# ──────────────────────────────────────────────────────────────────────

def get_hold_period_portfolio_return(
    conn: sqlite3.Connection,
    weights: Dict[str, float],
    start_date: str,
    end_date: str,
) -> float:
    """Return weighted cumulative % change of a basket between two dates."""
    weighted = []
    for tkr, w in weights.items():
        df = pd.read_sql_query(
            """
            SELECT date, close FROM price
            WHERE ticker = ? AND date BETWEEN ? AND ? ORDER BY date
            """,
            conn,
            params=(tkr, start_date, end_date),
        )
        if df.empty or len(df) < 2:
            continue
        ret = df.iloc[-1]["close"] / df.iloc[0]["close"] - 1.0
        weighted.append(ret * w)
    return float(np.sum(weighted)) if weighted else np.nan


def compute_var_cvar(rets: pd.Series, conf: float = 0.95) -> Tuple[float, float]:
    var = rets.quantile(1 - conf)
    cvar = rets[rets <= var].mean()
    return var, cvar


def hedge_allocation(var: float, cvar: float, threshold: float = -0.02) -> float:
    return min(1.0, abs(var) / abs(threshold)) if var < threshold else 0.0


def calculate_combined_return(equ_ret: float, cc_ret: float, w: float) -> float:
    return ((1 + equ_ret) ** w) * ((1 + cc_ret) ** (1 - w)) - 1

# ──────────────────────────────────────────────────────────────────────
#  Core driver
# ──────────────────────────────────────────────────────────────────────

def run_combined_strategy(
    *,
    start_date: str,
    end_date: str,
    capital: float,
    otm_delta: float,
    expiry_days: int,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float,
    commission_rate: float,
    vol_lookback: int,
    use_dynamic_vol: bool,
    steps: int,
    high_quantile: float,
    low_quantile: float,
    rebalance_frequency: int,
    roll_lookback: int,
    price_lookback: int,
    num_portfolios: int,
    output_dir: str,
    equity_weight: float,
):
    """Complete workflow: hybrid equity → covered calls → merge & stats."""

    conn = sqlite3.connect(DB_PATH)

    # ------------------------------------------------------------------
    # 1) Hybrid equity engine
    # ------------------------------------------------------------------
    logging.info("Running hybrid equity engine …")
    clear_output_table(conn)
    hybrid_engine(
        conn,
        start_date,
        end_date,
        high_quantile,
        low_quantile,
        rebalance_frequency,
        roll_lookback,
        price_lookback,
        num_portfolios,
        risk_free_rate,
        output_dir,
    )

    logging.info("Hybrid summary: %s", calculate_overall_portfolio_stats(conn, start_date, end_date))

    # ------------------------------------------------------------------
    # 2) Covered‑call overlay
    # ------------------------------------------------------------------
    logging.info("Simulating covered‑call overlay …")
    covered_call_sim(
        conn,
        start_date,
        end_date,
        capital,
        otm_delta,
        expiry_days,
        r=risk_free_rate,
        base_sigma=volatility,
        dividend_yield=dividend_yield,
        commission_rate=commission_rate,
        use_dynamic_vol=use_dynamic_vol,
        vol_lookback=vol_lookback,
        steps=steps,
    )
    # AFTER
    logging.info("Covered‑call summary: %s",
                 cc_portfolio_stats(conn, start_date, end_date))

    # ------------------------------------------------------------------
    # 3) Merge time‑series & plot
    # ------------------------------------------------------------------
    df_equ = pd.read_sql_query(
        """
        SELECT analysis_date, (portfolio_return+1) AS mult
        FROM optimized_hybrid_portfolios
        WHERE analysis_date BETWEEN ? AND ? ORDER BY analysis_date
        """,
        conn,
        params=(start_date, end_date),
    )
    df_cc = pd.read_sql_query(
        """
        SELECT analysis_date, (period_return+1) AS mult
        FROM portfolio_covered_calls
        WHERE analysis_date BETWEEN ? AND ? ORDER BY analysis_date
        """,
        conn,
        params=(start_date, end_date),
    )

    df_equ["analysis_date"] = pd.to_datetime(df_equ["analysis_date"])
    df_cc["analysis_date"] = pd.to_datetime(df_cc["analysis_date"])

    df_merged = pd.merge(df_equ, df_cc, on="analysis_date", suffixes=("_equ", "_cc"))
    df_merged["combined_mult"] = (
        equity_weight * df_merged["mult_equ"] + (1 - equity_weight) * df_merged["mult_cc"]
    )

    df_merged["cum_equ"] = df_merged["mult_equ"].cumprod()
    df_merged["cum_cc"] = df_merged["mult_cc"].cumprod()
    df_merged["cum_comb"] = df_merged["combined_mult"].cumprod()

    bench = fetch_benchmark_returns(conn, start_date, end_date, ticker="SPY")

    plt.figure(figsize=(11, 7))
    plt.plot(df_merged["analysis_date"], df_merged["cum_equ"], label="Equity portfolio")
    plt.plot(df_merged["analysis_date"], df_merged["cum_cc"], label="Covered calls")
    plt.plot(df_merged["analysis_date"], df_merged["cum_comb"], label="Combined")
    if not bench.empty:
        plt.plot(bench.index, bench.values, label="SPY benchmark")
    plt.title("Cumulative return comparison")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True, ls=":", lw=0.4)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(RESULTS_DIR) / "Equity_CoveredCalls_Combined_Returns.png", bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # 4) Risk metrics
    # ------------------------------------------------------------------
    df_val = pd.read_sql_query(
        """
        SELECT analysis_date, portfolio_value FROM portfolio_covered_calls
        WHERE analysis_date BETWEEN ? AND ? ORDER BY analysis_date
        """,
        conn,
        params=(start_date, end_date),
    )
    df_val["analysis_date"] = pd.to_datetime(df_val["analysis_date"])
    df_val["daily_ret"] = df_val["portfolio_value"].pct_change().fillna(0)

    var95, cvar95 = compute_var_cvar(df_val["daily_ret"], 0.95)
    logging.info("Combined VaR 95%%: %.4f  |  CVaR 95%%: %.4f", var95, cvar95)
    logging.info("Indicative hedge ratio (threshold -2%%): %.2f", hedge_allocation(var95, cvar95))

    conn.close()

# ──────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────

def cli():
    p = argparse.ArgumentParser("Hybrid portfolio + covered calls back‑tester")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--capital", type=float, default=100_000_000)
    p.add_argument("--otm_delta", type=float, default=0.05)
    p.add_argument("--expiry_days", type=int, default=30)
    p.add_argument("--risk_free_rate", type=float, default=0.02)
    p.add_argument("--volatility", type=float, default=0.25)
    p.add_argument("--dividend_yield", type=float, default=0.02)
    p.add_argument("--commission_rate", type=float, default=0.001)
    p.add_argument("--vol_lookback", type=int, default=30)
    p.add_argument("--use_dynamic_vol", action="store_true")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--high_quantile", type=float, default=0.90)
    p.add_argument("--low_quantile", type=float, default=0.0)
    p.add_argument("--rebalance_frequency", type=int, default=21)
    p.add_argument("--roll_lookback", type=int, default=21)
    p.add_argument("--price_lookback", type=int, default=126)
    p.add_argument("--num_portfolios", type=int, default=10_000)
    p.add_argument("--output_dir", default=str(RESULTS_DIR))
    p.add_argument("--equity_weight", type=float, default=0.5)
    args = p.parse_args()

    run_combined_strategy(
        start_date=args.start,
        end_date=args.end,
        capital=args.capital,
        otm_delta=args.otm_delta,
        expiry_days=args.expiry_days,
        risk_free_rate=args.risk_free_rate,
        volatility=args.volatility,
        dividend_yield=args.dividend_yield,
        commission_rate=args.commission_rate,
        vol_lookback=args.vol_lookback,
        use_dynamic_vol=args.use_dynamic_vol,
        steps=args.steps,
        high_quantile=args.high_quantile,
        low_quantile=args.low_quantile,
        rebalance_frequency=args.rebalance_frequency,
        roll_lookback=args.roll_lookback,
        price_lookback=args.price_lookback,
        num_portfolios=args.num_portfolios,
        output_dir=args.output_dir,
        equity_weight=args.equity_weight,
    )


if __name__ == "__main__":
    cli()
