#!/usr/bin/env python3
"""
Visuals Module – v2.2 (2025‑04‑16)

Generates a full performance‑analytics pack for the hybrid strategy:

• Monthly returns heat‑map
• Cumulative returns vs. SPY
• Drawdown comparison
• VaR / CVaR (daily histogram)
• VaR / CVaR (monthly):
      – SPY‑only
      – Strategy‑only
      – Overlay
• 30‑day rolling volatility
• 6‑month rolling correlation vs. SPY
• Collar‑hedged scenarios overlay
• Portfolio‑weights visuals
      – Classic heat‑map
      – Stream‑style stacked‑area (top‑10 + OTHER)
• NEW cumulative overlays
      – Equity · Covered Call · Combined · Avg‑Collar
      – Each Collar scenario vs. Equity & Combined

All PNGs are written to ``config.RESULTS_DIR``.  Missing tables or
data issues trigger warnings; the run continues.
"""
from __future__ import annotations

import calendar
import json
import logging
import os
import sqlite3
import sys
from typing import Callable, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ─── project‑root import ───────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config  # type: ignore  # noqa: E402

RESULTS = config.RESULTS_DIR
os.makedirs(RESULTS, exist_ok=True)

# ─── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s • %(levelname)s • %(message)s",
    datefmt="%Y‑%m‑%d %H:%M:%S",
)

# ─── helper utilities ──────────────────────────────────────────────────────────
def _tz_naive(s: pd.Series | pd.DataFrame):
    """Drop timezone info so indexes align cleanly."""
    if getattr(s.index, "tz", None):
        s.index = s.index.tz_convert(None)
    return s


def _safe_pct_change(s: pd.Series) -> pd.Series:
    return s.pct_change().dropna()


def _monthly_returns(cum: pd.Series) -> pd.Series:
    """Month‑end percentage returns from a cumulative series."""
    return cum.resample("M").last().pct_change().dropna()


def _save(fig: plt.Figure, name: str) -> None:
    out = os.path.join(RESULTS, name)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    logging.info("✅  Saved %s", out)


# ─── DATA FETCH ────────────────────────────────────────────────────────────────
def fetch_equity(conn: sqlite3.Connection) -> pd.Series:
    df = pd.read_sql(
        """SELECT analysis_date, portfolio_return
           FROM optimized_hybrid_portfolios
           ORDER BY analysis_date""",
        conn,
        parse_dates=["analysis_date"],
    ).set_index("analysis_date")
    return (1 + df["portfolio_return"]).cumprod().rename("equity")


def fetch_covered(conn: sqlite3.Connection) -> pd.Series:
    df = pd.read_sql(
        """SELECT analysis_date, period_return
           FROM portfolio_covered_calls
           ORDER BY analysis_date""",
        conn,
        parse_dates=["analysis_date"],
    ).set_index("analysis_date")
    return (1 + df["period_return"]).cumprod().rename("covered_call")


def fetch_combined(conn: sqlite3.Connection) -> pd.Series:
    eq, cc = fetch_equity(conn), fetch_covered(conn)
    df = pd.concat([eq, cc], axis=1, join="inner")
    w = getattr(config, "EQUITY_WEIGHT", 0.5)
    return ((df["equity"] ** w) * (df["covered_call"] ** (1 - w))).rename("combined")


def fetch_spy(conn: sqlite3.Connection) -> pd.Series:
    df = pd.read_sql(
        """SELECT date, close FROM price
           WHERE ticker='SPY' ORDER BY date""",
        conn,
        parse_dates=["date"],
    ).set_index("date")
    return (df["close"].pct_change().fillna(0).add(1).cumprod()).rename("SPY")


def fetch_collar(conn: sqlite3.Connection) -> pd.DataFrame:
    exists = pd.read_sql(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name='portfolio_collar_hedged'",
        conn,
    ).shape[0]
    if not exists:
        logging.warning("⚠️  No `portfolio_collar_hedged` table – skipping collar charts.")
        return pd.DataFrame()

    df = (
        pd.read_sql(
            """SELECT analysis_date, scenario, cumulative_return
               FROM portfolio_collar_hedged
               ORDER BY analysis_date""",
            conn,
            parse_dates=["analysis_date"],
        )
        .drop_duplicates(subset=["analysis_date", "scenario"])
        .sort_values("analysis_date")
    )

    return df.pivot_table(
        index="analysis_date",
        columns="scenario",
        values="cumulative_return",
        aggfunc="last",
    )


def fetch_weights(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql(
        """SELECT analysis_date, weights
           FROM optimized_hybrid_portfolios
           ORDER BY analysis_date""",
        conn,
        parse_dates=["analysis_date"],
    )
    records: List[dict] = []
    for row in df.itertuples(index=False):
        try:
            d = json.loads(row.weights)
        except Exception:
            d = {}
        d["analysis_date"] = row.analysis_date
        records.append(d)

    wdf = (
        pd.DataFrame.from_records(records)
        .set_index("analysis_date")
        .fillna(0)
        .sort_index()
    )
    return wdf.loc[:, (wdf != 0).any(axis=0)]  # drop all‑zero columns


# ─── PLOTTING functions (legacy) ───────────────────────────────────────────────
conn = sqlite3.connect(config.DB_PATH)
collar_df = fetch_collar(conn)
# choose your scenario key
scenario = "collar_floor0.995_cap1.03"
collar = collar_df[scenario]
collar.name = "Collar‑Hedged"
def plot_monthly_heatmap(combined: pd.Series) -> None:
    monthly = combined.resample("M").last().pct_change().dropna() * 100
    df = pd.DataFrame(
        {"Year": monthly.index.year, "Month": monthly.index.month, "Return": monthly}
    )
    pivot = df.pivot(index="Year", columns="Month", values="Return").reindex(
        columns=range(1, 13)
    )
    pivot.columns = [calendar.month_abbr[m] for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        cbar_kws={"label": "% Return"},
        ax=ax,
    )
    ax.set_title("Monthly Returns Heatmap")
    _save(fig, "plot_monthly_heatmap.png")

def plot_covered_calls_cumulative_return(
    equity: pd.Series,
    covered: pd.Series,
    spy: pd.Series):


    fig, ax = plt.subplots(figsize=(11, 5))
    for s, lbl in [
        (equity,  "Equity Portfolio"),
        (covered, "Equity + Covered Calls Portfolio"),
        (spy,     "SPY"),
    ]:
        _tz_naive(s).plot(ax=ax, label=lbl, linewidth=2)

    ax.set(title="Cumulative Returns (Growth of $1)", ylabel="Value")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    _save(fig, "plot_covered_calls_cumulative_returns.png")

def plot_cumulative_returns(
    equity: pd.Series,
    covered: pd.Series,
    combined: pd.Series,
    spy: pd.Series,
    collar_scenario: str = "collar_floor0.995_cap1.03",
) -> None:
    # open your own DB connection
    import sqlite3
    conn = sqlite3.connect(config.DB_PATH)
    try:
        df_collar = fetch_collar(conn)
    finally:
        conn.close()

    # normalize the collar series if it exists
    collar = None
    if collar_scenario in df_collar:
        series = df_collar[collar_scenario].dropna()
        collar = series / series.iloc[0]
    else:
        logging.warning("Scenario %s not found in collar table", collar_scenario)

    fig, ax = plt.subplots(figsize=(11, 5))
    for s, lbl in [
        (equity,  "Equity Portfolio"),
        (covered, "Equity + Covered Calls Portfolio"),
        (spy,     "SPY"),
    ]:
        _tz_naive(s).plot(ax=ax, label=lbl, linewidth=2)

    if collar is not None:
        _tz_naive(collar).plot(
            ax=ax,
            label=f"Collar Hedged ({collar_scenario})",
            linestyle="--",
            linewidth=2.5,
        )

    ax.set(title="Cumulative Returns (Growth of $1)", ylabel="Value")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    _save(fig, "plot_cumulative_returns.png")



def plot_drawdown(
    equity: pd.Series,
    covered: pd.Series,
    combined: pd.Series,
    spy: pd.Series,
    collar_scenario: str = "collar_floor0.995_cap1.03",
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for s, lbl in [
        (equity, "Equity"),
        (covered, "Covered Call"),
        (combined, "Combined"),
        (spy, "SPY"),
    ]:
        dd = _tz_naive(s) / _tz_naive(s).cummax() - 1
        ax.plot(dd, label=lbl, linewidth=1.5)

    # fetch and plot collar drawdown
    import sqlite3
    conn2 = sqlite3.connect(config.DB_PATH)
    try:
        df_collar = fetch_collar(conn2)
    finally:
        conn2.close()

    if collar_scenario in df_collar:
        raw = df_collar[collar_scenario].dropna()
        collar = raw / raw.iloc[0]
        dd_collar = collar / collar.cummax() - 1
        ax.plot(
            _tz_naive(dd_collar),
            label=f"Collar Hedged ({collar_scenario})",
            linestyle="--",
            linewidth=2,
        )
    else:
        logging.warning("Scenario %s not found; skipping collar drawdown.", collar_scenario)

    ax.set(title="Drawdown", ylabel="Drawdown")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    _save(fig, "plot_drawdown.png")



def plot_var_cvar_daily(combined: pd.Series, spy: pd.Series) -> None:
    r1, r2 = _safe_pct_change(combined), _safe_pct_change(spy)
    if r1.empty or r2.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(r1, bins=50, alpha=0.6, label="Strategy")
    ax.hist(r2, bins=50, alpha=0.4, label="SPY")

    for r, c in [(r1, "tab:blue"), (r2, "tab:orange")]:
        var = r.quantile(0.05)
        cvar = r[r <= var].mean()
        ax.axvline(var, color=c, linestyle="-", linewidth=2)
        ax.axvline(cvar, color=c, linestyle="--", linewidth=2)

    ax.set(title="VaR / CVaR – Daily Returns", xlabel="Daily Return")
    ax.legend()
    _save(fig, "plot_var_cvar_daily.png")


def plot_rolling_vol(combined: pd.Series, spy: pd.Series) -> None:
    d1, d2 = _safe_pct_change(combined), _safe_pct_change(spy)
    df = pd.concat([d1, d2], axis=1, join="inner").dropna()
    if df.empty:
        return
    rv = df.rolling(window=30, min_periods=10).std() * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(12, 4))
    rv.iloc[:, 0].plot(ax=ax, label="Strategy")
    rv.iloc[:, 1].plot(ax=ax, label="SPY", linestyle="--")

    ax.set(title="30‑Day Rolling Volatility (Annualized)", ylabel="Volatility")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    _save(fig, "plot_rolling_vol.png")


def plot_rolling_corr(combined: pd.Series, spy: pd.Series) -> None:
    d1, d2 = _safe_pct_change(combined), _safe_pct_change(spy)
    df = pd.concat([d1, d2], axis=1, join="inner").dropna()
    if len(df) < 30:
        return
    corr = df.iloc[:, 0].rolling(window=min(126, len(df)), min_periods=30).corr(
        df.iloc[:, 1]
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    corr.plot(ax=ax, label="6‑Month Corr")

    ax.set(title="6‑Month Rolling Correlation vs SPY", ylabel="Correlation")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    _save(fig, "plot_rolling_corr.png")


def plot_collar_scenarios(collar_df: pd.DataFrame, spy: pd.Series) -> None:
    if collar_df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    for col in collar_df.columns:
        ser = collar_df[col].dropna()
        ax.plot(ser / ser.iloc[0], label=col)

    ax.plot(spy / spy.iloc[0], "--", label="SPY")
    ax.set_title("Collar‑Hedged Scenarios vs SPY")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    _save(fig, "plot_collar_scenarios.png")


def plot_weights_heatmap(weights_df: pd.DataFrame) -> None:
    if weights_df.empty:
        return
    fig, ax = plt.subplots(figsize=(13, 8))
    sns.heatmap(
        weights_df.T,
        cmap="viridis",
        cbar_kws={"label": "Weight"},
        mask=weights_df.T == 0,
        ax=ax,
    )
    ax.set_title("Portfolio Weights Over Time")
    plt.xticks(rotation=45, ha="right")
    _save(fig, "plot_weights_heatmap.png")


# ─── Monthly VaR/CVaR & weights stream ─────────────────────────────────────────
def _plot_var_cvar(ax: plt.Axes, r: pd.Series, color: str, label: str) -> None:
    """
    Plot a histogram of returns and annotate 95% VaR and CVaR lines.
    """
    # Compute VaR and CVaR
    var = r.quantile(0.05)
    cvar = r[r <= var].mean()

    # Plot histogram of returns
    ax.hist(r, bins=40, alpha=0.6, color=color, label=f"{label} Returns")

    # Draw and label VaR line
    ax.axvline(
        var,
        color=color,
        linestyle="-",
        linewidth=2,
        label=f"{label} 95% VaR: {var:.1%}"
    )

    # Draw and label CVaR line
    ax.axvline(
        cvar,
        color=color,
        linestyle="--",
        linewidth=2,
        label=f"{label} 95% CVaR: {cvar:.1%}"
    )


def plot_var_cvar_monthly_overlay(combined: pd.Series, spy: pd.Series) -> None:
    """
    Plot monthly return distributions for the collar‑hedged portfolio vs. SPY,
    with annotated 95% VaR and CVaR lines.
    """
    # Calculate month‑end percentage returns
    m1 = combined.resample("M").last().pct_change().dropna()
    m2 = spy.resample("M").last().pct_change().dropna()
    if m1.empty or m2.empty:
        return

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(9, 4))

    # Plot both distributions with VaR/CVaR annotations
    _plot_var_cvar(ax, m1, "tab:blue",   "Collar Hedged Monthly")
    _plot_var_cvar(ax, m2, "tab:orange", "SPY Monthly")

    # Set titles and axis labels
    ax.set_title("Monthly VaR / CVaR – Collar Hedged vs SPY")
    ax.set_xlabel("Monthly Return")
    ax.set_ylabel("Frequency")

    # Display combined legend and grid
    ax.legend(loc="best", fontsize="small")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Save the figure
    _save(fig, "plot_var_cvar_monthly_overlay.png")



def plot_weights_stream(weights_df: pd.DataFrame, top_n: int = 10) -> None:
    if weights_df.empty:
        return
    top_assets = weights_df.mean().nlargest(top_n).index.tolist()
    main = weights_df[top_assets]
    other = 1 - main.sum(axis=1)
    main["OTHER"] = other.clip(lower=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(main.index, main.T, labels=main.columns)
    ax.set(title=f"Portfolio Weights Stream (Top {top_n})", ylabel="Weight")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y‑%m"))
    plt.xticks(rotation=45, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    _save(fig, "plot_weights_stream.png")


# ─── NEW cumulative overlays ───────────────────────────────────────────────────
def plot_cumulative_baselines(
    equity: pd.Series,
    covered: pd.Series,
    combined: pd.Series,
    collar_df: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for s, lbl, lw in [
        (equity, "Equity", 1.8),
        (covered, "Covered Call", 1.8),
        (combined, "Combined", 2.2),
    ]:
        _tz_naive(s).plot(ax=ax, label=lbl, linewidth=lw)

    if not collar_df.empty:
        avg_collar = collar_df.mean(axis=1).dropna()
        (avg_collar / avg_collar.iloc[0]).plot(
            ax=ax, label="Collar (avg)", linestyle="--", linewidth=2.2
        )

    ax.set(title="Cumulative Performance – Baselines", ylabel="Growth of $1")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    _save(fig, "plot_cumulative_baselines.png")


def plot_collar_vs_baselines(
    combined: pd.Series, equity: pd.Series, collar_df: pd.DataFrame
) -> None:
    if collar_df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    _tz_naive(equity).plot(ax=ax, label="Equity", linewidth=1.5)
    _tz_naive(combined).plot(ax=ax, label="Combined", linewidth=2.0)

    for col in collar_df.columns:
        ser = collar_df[col].dropna()
        (ser / ser.iloc[0]).plot(
            ax=ax, linewidth=0.9, alpha=0.7, label=f"Collar: {col}"
        )

    ax.set(title="Collar Scenarios vs. Baselines", ylabel="Growth of $1")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(ncol=2, fontsize="small")
    _save(fig, "plot_collar_vs_baselines.png")

def plot_rolling_sharpe(
    collar: pd.Series, combined: pd.Series, spy: pd.Series,
    window: int = 12, rf_annual: float = 0.02
) -> None:
    """
    Plot a rolling (annualized) Sharpe ratio for:
      - the collar‑hedged portfolio,
      - the combined (70/30) strategy,
      - SPY.
    Uses a rolling window of `window` months and annual risk-free rate rf_annual.
    """
    # 1) compute month‑end returns
    ret_collar   = collar.resample("M").last().pct_change().dropna()
    ret_combined = combined.resample("M").last().pct_change().dropna()
    ret_spy      = spy.resample("M").last().pct_change().dropna()

    # 2) align into DataFrame
    df = pd.concat({
        "Collar‑Hedged Monthly Sharpe":   ret_collar,
        "Combined Equity + Covered Calls Monthly Sharpe": ret_combined,
        "SPY Monthly Sharpe":             ret_spy
    }, axis=1).dropna()

    # 3) convert annual RF to monthly
    rf_monthly = (1 + rf_annual) ** (1/12) - 1

    # 4) excess returns
    excess = df.subtract(rf_monthly, axis=0)

    # 5) rolling statistics
    roll_mean = excess.rolling(window).mean()
    roll_std  = excess.rolling(window).std()

    # 6) compute annualized Sharpe
    rolling_sharpe = (roll_mean / roll_std) * np.sqrt(12)

    # 7) plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in rolling_sharpe:
        ax.plot(rolling_sharpe.index, rolling_sharpe[col], lw=2, label=col)

    ax.set_title(f"{window}‑Month Rolling Sharpe Ratio")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe Ratio (annualized)")
    ax.axhline(0, color="black", linestyle=":", linewidth=1)
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.6)

    _save(fig, "plot_rolling_sharpe.png")

# ─── MAIN driver ───────────────────────────────────────────────────────────────
def main() -> None:
    conn = sqlite3.connect(config.DB_PATH)
    try:
        equity = fetch_equity(conn)
        covered = fetch_covered(conn)
        combined = fetch_combined(conn)
        spy = fetch_spy(conn)
        collar_df = fetch_collar(conn)
        weights_df = fetch_weights(conn)
    finally:
        conn.close()

    tasks: List[Callable[[], None]] = [
        # legacy / v2.1 charts
        lambda: plot_monthly_heatmap(collar),
        lambda: plot_cumulative_returns(equity, covered, combined, spy, "collar_floor0.995_cap1.03"),
        lambda: plot_covered_calls_cumulative_return(equity, covered, spy),
        lambda: plot_drawdown(equity, covered, combined, spy),
        lambda: plot_var_cvar_daily(combined, spy),
        lambda: plot_rolling_vol(combined, spy),
        lambda: plot_rolling_corr(combined, spy),
        lambda: plot_collar_scenarios(collar_df, spy),
        lambda: plot_weights_heatmap(weights_df),
        lambda: plot_var_cvar_monthly_overlay(collar, spy),
        lambda: plot_weights_stream(weights_df, top_n=10),
        lambda: plot_rolling_sharpe(collar, combined, spy),
        # new overlays
        lambda: plot_cumulative_baselines(equity, covered, combined, collar_df),
        lambda: plot_collar_vs_baselines(combined, equity, collar_df),
    ]

    for fn in tasks:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 (broad but logged)
            logging.exception("❌  Error in %s: %s", fn.__name__, exc)


if __name__ == "__main__":
    main()

