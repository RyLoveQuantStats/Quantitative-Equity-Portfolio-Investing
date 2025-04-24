#!/usr/bin/env python3
"""
100% Equity + Covered Calls vs. 100% Equity + Collar Overlay
-------------------------------------------------------------
This script:
 1. Fetches equity-only returns
 2. Builds 100% equity cumulative value
 3. Builds 100% equity + covered-calls cumulative value
 4. Builds 100% equity + collar cumulative value
 5. Computes rolling realized volatility
 6. Prices American calls & puts via CRR tree with early exercise
 7. Applies volatility skew to puts
 8. Clamps call strikes to your cap multiplier
 9. Adjusts underlying for discrete dividends
10. Applies bid/ask spread, slippage & commissions
11. Generates:
    - Growth of $1 for Equity, Calls, Collar, SPY
    - Drawdown comparison
    - 12-month rolling Sharpe
    - VaR95 & CVaR95
    - Monthly returns heatmap for collar

    python -m scripts.collar_hedging --start 2016-01-01 --end 2024-12-31 --floor 0.97 --cap 1.05 --expiry_days 30 --vol_lookback 21 --r 0.02 --q 0.0 --commission 0.001 --steps 200 --skew 0.10 --div_dates 2020-06-15,2021-06-15 --div_amounts 0.5,0.5 --spread_pct 0.005 --slippage_pct 0.001 --initial-capital 100000000
"""
import os, logging, argparse, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from math import exp, sqrt
from datetime import datetime
import matplotlib.pyplot as plt
from scripts.combined_strategy_rmw_calls import fetch_benchmark_returns, compute_var_cvar
import scripts.config as config
import yfinance as yf
from math import sqrt
from datetime import timedelta

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
DB_PATH = Path('database/data.db')

def binomial_option_price_american(S0, K, T, r, sigma, q, steps, option_type, div_schedule=None):
    if div_schedule:
        for ex_date, amt in div_schedule:
            t = (ex_date - datetime.now()).days / 252
            if t > 0:
                S0 -= amt * exp(-r * t)
    dt   = T/steps
    u    = exp(sigma * sqrt(dt))
    d    = 1/u
    disc = exp(-r * dt)
    p    = (exp((r - q) * dt) - d) / (u - d)

    # underlying tree
    ST = np.zeros((steps+1, steps+1))
    ST[0,0] = S0
    for i in range(1, steps+1):
        ST[i,0] = ST[i-1,0] * d
        for j in range(1, i+1):
            ST[i,j] = ST[i-1,j-1] * u

    # payoff at expiry
    V = np.zeros_like(ST)
    if option_type == 'call':
        V[steps] = np.maximum(ST[steps] - K, 0)
    else:
        V[steps] = np.maximum(K - ST[steps], 0)

    # backward induction
    for i in range(steps-1, -1, -1):
        for j in range(i+1):
            cont = disc * (p * V[i+1,j+1] + (1-p) * V[i+1,j])
            exer = (ST[i,j]-K) if option_type=='call' else (K-ST[i,j])
            V[i,j] = max(cont, exer)
    return V[0,0]

def dynamic_otm_delta(vol, low_vol=0.15, high_vol=0.25, base_delta=0.05, prot_delta=0.10):
    if vol <= low_vol:   return base_delta
    if vol >= high_vol:  return prot_delta
    frac = (vol - low_vol) / (high_vol - low_vol)
    return base_delta + frac * (prot_delta - base_delta)

def calculate_equity(conn, start, end, initial_capital):
    df = pd.read_sql_query(
        "SELECT analysis_date, portfolio_return "
        "FROM optimized_hybrid_portfolios "
        "WHERE analysis_date BETWEEN ? AND ? "
        "ORDER BY analysis_date",
        conn, params=(start,end)
    )
    df['analysis_date']  = pd.to_datetime(df['analysis_date'])
    df = df.sort_values('analysis_date')
    df['equity_return'] = pd.to_numeric(df['portfolio_return'], errors='coerce')

    # start with INITIAL_CAPITAL and grow it
    df['equity_value']  = initial_capital * (1 + df['equity_return']).cumprod()
    return df[['analysis_date','equity_return','equity_value']]


def apply_covered_calls(df,
                        cap,
                        expiry_days,
                        r, q,
                        commission,
                        steps,
                        vol_lookback,    # kept for signature
                        skew,            # not used here but kept
                        div_schedule,
                        spread_pct,
                        slippage_pct):
    """
    Sells calls every expiry_days using VIX-based vol, collects premiums,
    builds cash leg + covered portfolio, and flags rebalance dates.
    """
    df = df.copy().reset_index(drop=True)
    df['analysis_date'] = pd.to_datetime(df['analysis_date'])

    # use VIX instead of realized vol
    df['vix_vol']      = df['vix'] / 100.0
    df['rebalance']    = False

    # placeholders
    df['premium_cash']  = 0.0
    df['cash_value']    = 0.0
    df['covered_value'] = 0.0

    T         = expiry_days / 252
    sale_idxs = np.arange(0, len(df), expiry_days)

    for i in sale_idxs:
        vol = df.at[i, 'vix_vol']
        if np.isnan(vol):
            continue

        S0  = df.at[i, 'equity_value']
        eff = dynamic_otm_delta(vol)
        raw_Kc = S0 * (1 + (cap - 1) * (eff / 0.05))
        Kc     = min(raw_Kc, S0 * cap)

        C_mid = binomial_option_price_american(
            S0, Kc, T, r, vol, q, steps, 'call', div_schedule
        )
        C_tr = C_mid * (1 + spread_pct/2)
        slp  = slippage_pct * C_mid
        comm = commission * C_tr

        df.at[i, 'premium_cash'] = C_tr - slp - comm
        df.at[i, 'rebalance']    = True

    # build cash & covered-value legs
    for i in range(len(df)):
        prev_cash = df.at[i-1, 'cash_value'] if i>0 else 0.0
        cash = prev_cash + df.at[i, 'premium_cash']
        df.at[i, 'cash_value']    = cash
        df.at[i, 'covered_value'] = df.at[i, 'equity_value'] + cash

    df['covered_return'] = df['covered_value'].pct_change().fillna(0)
    return df[['analysis_date',
               'equity_return','equity_value',
               'covered_return','covered_value',
               'rebalance']]


def apply_collar(df,
                 floor, cap, expiry_days,
                 r, q, commission, steps,
                 vol_lookback, skew, div_schedule,
                 spread_pct, slippage_pct):
    """
    Buys puts and sells calls every expiry_days using VIX-based vol,
    collects net premium, schedules payoffs at expiry, flags rebalance,
    then builds cash leg + collar portfolio.
    """
    df = df.copy().reset_index(drop=True)
    df['analysis_date'] = pd.to_datetime(df['analysis_date'])

    # use VIX instead of realized vol
    df['vix_vol']      = df['vix'] / 100.0
    df['rebalance']    = False

    # placeholders
    df['net_premium']  = 0.0
    df['cash_value']   = 0.0
    df['collar_value'] = 0.0
    df['Kc']           = np.nan
    df['Kf']           = np.nan

    # determine roll indices by stepping expiry_days in calendar days
    start = df['analysis_date'].iloc[0]
    end   = df['analysis_date'].iloc[-1]
    roll_dates = []
    cur = start
    while cur <= end:
        idx = df['analysis_date'].searchsorted(cur)
        if idx < len(df):
            roll_dates.append(idx)
        cur += timedelta(days=expiry_days)

    T = expiry_days / 252

    # 1) collect up-front net premium & store strikes
    for i in roll_dates:
        vol = df.at[i, 'vix_vol']
        if np.isnan(vol):
            continue

        S0  = df.at[i, 'equity_value']
        eff    = dynamic_otm_delta(vol)
        raw_Kc = S0 * (1 + (cap - 1) * (eff / 0.05))
        Kc     = min(raw_Kc, S0 * cap)
        Kf     = S0 * floor

        C_mid = binomial_option_price_american(
            S0, Kc, T, r, vol, q, steps, 'call', div_schedule
        )
        P_mid = binomial_option_price_american(
            S0, Kf, T, r, vol * (1 + skew), q, steps, 'put', div_schedule
        )

        C_tr = C_mid * (1 + spread_pct/2)
        P_tr = P_mid * (1 - spread_pct/2)
        slp  = slippage_pct * (C_mid + P_mid)
        comm = commission * (C_tr + P_tr)

        df.at[i, 'net_premium'] = P_tr - C_tr - slp - comm
        df.at[i, 'Kc']          = Kc
        df.at[i, 'Kf']          = Kf
        df.at[i, 'rebalance']   = True

    # 2) schedule exercise payoffs at each expiry
    for i in roll_dates:
        expiry_date = df.at[i, 'analysis_date'] + timedelta(days=expiry_days)
        j = df['analysis_date'].searchsorted(expiry_date)
        if j >= len(df):
            continue
        S_end = df.at[j, 'equity_value']
        Kc    = df.at[i, 'Kc']
        Kf    = df.at[i, 'Kf']
        if np.isnan(Kc) or np.isnan(Kf):
            continue

        put_payoff  = max(Kf - S_end, 0)
        call_payoff = -max(S_end - Kc, 0)
        df.at[j, 'net_premium'] += put_payoff + call_payoff

    # 3) build cash & collar legs
    for i in range(len(df)):
        prev_cash = df.at[i-1, 'cash_value'] if i>0 else 0.0
        cash = prev_cash + df.at[i, 'net_premium']
        df.at[i,  'cash_value']   = cash
        df.at[i,  'collar_value'] = df.at[i, 'equity_value'] + cash

    df['collar_return'] = df['collar_value'].pct_change().fillna(0)
    return df[['analysis_date','collar_return','collar_value','rebalance']]

def plot_growth(df, bench, out_dir):
    """
    Plots:
      • Growth of $1 for Equity, Calls, Collar, SPY in top pane
      • VIX index in the bottom pane
    """
    import os
    import matplotlib.pyplot as plt

    # expect a 'vix' column in df
    df = df.set_index('analysis_date').copy()

    # --- top pane: cumulative growth ---
    eo = df['equity_value'] / df['equity_value'].iloc[0]
    cc = df['covered_value'] / df['covered_value'].iloc[0]
    cl = df['collar_value']  / df['collar_value'].iloc[0]

    # --- create subplots ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        sharex=True,
        figsize=(12, 8),
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # top: growth of $1
    ax1.plot(eo, label='Equity Only ($1)')
    ax1.plot(cc, label='Equity+Calls ($1)')
    ax1.plot(cl, label='Equity+Collar ($1)')
    if bench is not None and not bench.empty:
        spy = bench.reindex(df.index).dropna()
        spy = spy / spy.iloc[0]
        ax1.plot(spy, '--k', label='SPY ($1)')
    ax1.set_title("Growth of $1")
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # bottom: VIX index
    if 'vix' in df.columns:
        ax2.plot(df['vix'], label='VIX')
    else:
        ax2.text(0.5, 0.5, "No VIX data available",
                 ha='center', va='center', transform=ax2.transAxes)
    ax2.set_ylabel('VIX')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "collar_growth_with_vix.png"))
    plt.close(fig)


def compute_drawdowns(cum):
    peak = cum.cummax()
    return cum/peak - 1


def plot_drawdowns(df, bench, out_dir):
    df = df.set_index('analysis_date')
    dd_eq = compute_drawdowns(df['equity_value']/df['equity_value'].iloc[0])
    dd_cc = compute_drawdowns(df['covered_value']/df['covered_value'].iloc[0])
    dd_cl = compute_drawdowns(df['collar_value']/df['collar_value'].iloc[0])

    plt.figure(figsize=(12,6))
    plt.plot(dd_eq, label='Equity')
    plt.plot(dd_cc, label='Equity+Calls')
    plt.plot(dd_cl, label='Equity+Collar')
    if bench is not None and not bench.empty:
        spy = bench.reindex(df.index).dropna()
        dd_spy = compute_drawdowns(spy/spy.iloc[0])
        plt.plot(dd_spy,'--k',label='SPY')
    plt.title("Drawdowns")
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"collar_drawdowns.png"))
    plt.close()


def rolling_sharpe(returns, window=12, rf=0.02):
    def _sr(x):
        ann_r = x.mean()*12
        ann_v = x.std()*sqrt(12)
        return (ann_r-rf)/ann_v if ann_v>0 else np.nan
    return returns.rolling(window).apply(_sr, raw=False)


def plot_rolling_sharpe(df, bench, out_dir):
    df = df.set_index('analysis_date')
    r_eq = df['equity_value'].pct_change().dropna()
    r_cc = df['covered_value'].pct_change().dropna()
    r_cl = df['collar_value'].pct_change().dropna()
    r_sp = bench.reindex(df.index).pct_change().dropna() if bench is not None else pd.Series()

    sr_eq = rolling_sharpe(r_eq)
    sr_cc = rolling_sharpe(r_cc)
    sr_cl = rolling_sharpe(r_cl)
    sr_sp = rolling_sharpe(r_sp) if not r_sp.empty else None

    plt.figure(figsize=(12,6))
    plt.plot(sr_eq, label='Equity')
    plt.plot(sr_cc, label='Calls')
    plt.plot(sr_cl, label='Collar')
    if sr_sp is not None: plt.plot(sr_sp,'--k',label='SPY')
    plt.title("12‑Month Rolling Sharpe")
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"collar_sharpe.png"))
    plt.close()


def var_comparison(df, bench):
    df = df.set_index('analysis_date')
    for name, series in [
        ('Equity', df['equity_value'].pct_change().dropna()),
        ('Equity+Calls', df['covered_value'].pct_change().dropna()),
        ('Equity+Collar',df['collar_value'].pct_change().dropna()),
        ('SPY', bench.reindex(df.index).pct_change().dropna() if bench is not None else pd.Series())
    ]:
        if series.empty: continue
        v,c = compute_var_cvar(series,0.95)
        logging.info(f"{name} VaR95={v:.4f}, CVaR95={c:.4f}")

def plot_var_cvar_comparison(df, bench, out_dir):
    # build all four return series
    df = df.set_index('analysis_date')
    series = {
        'Equity':        df['equity_value'].pct_change().dropna(),
        'Equity+Calls':  df['covered_value'].pct_change().dropna(),
        'Equity+Collar': df['collar_value'].pct_change().dropna()
    }
    if bench is not None and not bench.empty:
        series['SPY'] = bench.reindex(df.index).pct_change().dropna()

    # compute VaR95 and CVaR95 for each
    var95  = {}
    cvar95 = {}
    for name, ret in series.items():
        v, c = compute_var_cvar(ret, 0.95)
        # flip sign so we plot positive bars for losses
        var95[name]  = -v
        cvar95[name] = -c

    stats = pd.DataFrame({'VaR95': var95, 'CVaR95': cvar95})

    # plot side‑by‑side bars
    ax = stats.plot.bar(figsize=(8, 6))
    ax.set_ylabel('Loss at 95% confidence level')
    ax.set_title('VaR95 and CVaR95 Comparison')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'var_cvar_comparison.png'))
    plt.close()


def plot_monthly_heatmap(df, value_col, out_dir):
    """
    df         : DataFrame with an 'analysis_date' column, a numeric value_col,
                 and a boolean 'rebalance' flag.
    value_col  : e.g. 'covered_value' or 'collar_value'
    out_dir    : where to save the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime
    import os

    # prepare
    df = df.set_index('analysis_date').copy()
    df['ret'] = df[value_col].pct_change()
    df['year'], df['month'] = df.index.year, df.index.month

    # pivot for returns
    pivot = df.pivot_table(index='year', columns='month', values='ret')

    # find last rebalance per (year,month)
    reb = df[df['rebalance']]
    reb['year'], reb['month'] = reb.index.year, reb.index.month
    last_reb = reb.groupby(['year','month']).apply(lambda g: g.index.max())

    # plot heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot, aspect='auto', origin='lower', cmap='RdYlGn')
    fig.colorbar(im, ax=ax, label='Monthly Return')

    ax.set_xticks(np.arange(12))
    ax.set_xticklabels([datetime(1900,m,1).strftime('%b') for m in range(1,13)])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(int))
    ax.set_title(f"{value_col} Monthly Returns & Rebalance Dates")

    # annotate
    for i, year in enumerate(pivot.index):
        for j, month in enumerate(pivot.columns):
            val = pivot.loc[year, month]
            if not np.isnan(val):
                # return
                ax.text(j, i+0.2, f"{val*100: .1f}%",
                        ha='center', va='center', fontsize=8, color='black')
                # rebalance date
                dt = last_reb.get((year, month))
                if pd.notnull(dt):
                    ax.text(j, i-0.2, dt.strftime('%Y-%m-%d'),
                            ha='center', va='center', fontsize=6, color='black')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{value_col}_heatmap.png"))
    plt.close(fig)

def main():
    import argparse, sqlite3, logging, os
    from datetime import datetime
    import yfinance as yf
    import pandas as pd

    p = argparse.ArgumentParser(
        description="Backtest equity vs. covered calls vs. dynamic VIX‐collar"
    )
    # 1) CLI flags
    p.add_argument('--start',           type=str,   required=True, help="YYYY-MM-DD start date")
    p.add_argument('--end',             type=str,   required=True, help="YYYY-MM-DD end date")
    p.add_argument('--floor',           type=float, required=True, help="Put strike floor (e.g. 0.97)")
    p.add_argument('--cap',             type=float, required=True, help="Call strike cap (e.g. 1.05)")
    p.add_argument('--expiry_days',     type=int,   default=30,  help="Days to expiration")
    p.add_argument('--vol_lookback',    type=int,   default=21,  help="Unused when using VIX, but kept for signature")
    p.add_argument('--r',               type=float, default=0.02, help="Risk-free rate")
    p.add_argument('--q',               type=float, default=0.0,  help="Dividend yield")
    p.add_argument('--commission',      type=float, default=0.001,help="Commission rate")
    p.add_argument('--steps',           type=int,   default=200,  help="Binomial tree steps")
    p.add_argument('--skew',            type=float, default=0.10, help="Vol skew for puts")
    p.add_argument('--div_dates',       type=str,   default=None, help="Comma-sep ISO dates for dividends")
    p.add_argument('--div_amounts',     type=str,   default=None, help="Comma-sep amounts for each date")
    p.add_argument('--spread_pct',      type=float, default=0.005,help="Option half-spread (pct)")
    p.add_argument('--slippage_pct',    type=float, default=0.001,help="Slippage (pct of mid)")
    p.add_argument('--initial-capital', type=float, default=1e8,  help="Starting capital")

    args = p.parse_args()

    # 2) build dividend schedule
    div_schedule = []
    if args.div_dates and args.div_amounts:
        dates   = args.div_dates.split(',')
        amounts = list(map(float, args.div_amounts.split(',')))
        div_schedule = [
            (datetime.fromisoformat(d), a)
            for d, a in zip(dates, amounts)
        ]

    # --------------------------------------------------------------------------
    #  All DB work
    # --------------------------------------------------------------------------
    with sqlite3.connect(str(DB_PATH)) as conn:
        # 1) pure equity series
        df_eq = calculate_equity(conn,
                                 args.start,
                                 args.end,
                                 args.initial_capital)

        # 2) download VIX via Ticker.history (guarantees a "Close" column)
        vix_hist = yf.Ticker("^VIX").history(
            start=args.start,
            end=args.end,
            auto_adjust=False
        )

        # --- strip off any tz info so merge won’t fail ---
        if vix_hist.index.tz is not None:
            vix_hist.index = vix_hist.index.tz_localize(None)

        if "Close" not in vix_hist.columns:
            raise ValueError("VIX history missing 'Close' column")

        # 3) prepare vix DataFrame
        vix = (
            vix_hist["Close"]
               .reset_index()
               .rename(columns={"Date": "analysis_date", "Close": "vix"})
        )

        # 4) merge & forward-fill into your equity path
        df_eq = df_eq.merge(vix, on="analysis_date", how="left")
        df_eq["vix"] = df_eq["vix"].ffill()

        # 5) build covered calls & collars
        df_cc = apply_covered_calls(
            df_eq,
            cap          = args.cap,
            expiry_days  = args.expiry_days,
            r            = args.r,
            q            = args.q,
            commission   = args.commission,
            steps        = args.steps,
            vol_lookback = args.vol_lookback,
            skew         = args.skew,
            div_schedule = div_schedule,
            spread_pct   = args.spread_pct,
            slippage_pct = args.slippage_pct
        )
        df_cl = apply_collar(
            df_eq,
            floor        = args.floor,
            cap          = args.cap,
            expiry_days  = args.expiry_days,
            r            = args.r,
            q            = args.q,
            commission   = args.commission,
            steps        = args.steps,
            vol_lookback = args.vol_lookback,
            skew         = args.skew,
            div_schedule = div_schedule,
            spread_pct   = args.spread_pct,
            slippage_pct = args.slippage_pct
        )

        # 6) merge for combined performance
        df_h = pd.merge(df_cc, df_cl, on="analysis_date", how="inner")
        df_h = df_h.merge(
            df_eq[['analysis_date','vix']],
            on="analysis_date",
            how="left"
        )  
        # 7) fetch SPY benchmark
        try:
            bench = fetch_benchmark_returns(conn, args.start, args.end, ticker="SPY")
        except Exception:
            bench = None
            logging.warning("SPY fetch failed")

    # --------------------------------------------------------------------------
    #  plotting (after DB closed)
    # --------------------------------------------------------------------------
    plot_growth(df_h, bench, config.RESULTS_DIR)
    plot_drawdowns(df_h, bench, config.RESULTS_DIR)
    plot_rolling_sharpe(df_h, bench, config.RESULTS_DIR)
    plot_monthly_heatmap(df_cc, value_col="covered_value", out_dir=config.RESULTS_DIR)
    plot_monthly_heatmap(df_cl, value_col="collar_value",  out_dir=config.RESULTS_DIR)
    var_comparison(df_h, bench)
    plot_var_cvar_comparison(df_h, bench, config.RESULTS_DIR)


if __name__ == '__main__':
    main()