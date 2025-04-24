# Dynamic VIX-Driven Collar & Covered-Call Overlay Equity Portfolio

**Authors:** Ryan Loveless & Jaxson Fryer  
**Date:** April 2025  

---

## Project Overview

This repository implements a systematic quantitative equity strategy on Nasdaq-100 constituents, enhanced by two complementary options overlays: a covered-call write and a dynamic collar hedge. The core equity sleeve is constructed via factor-based screening (β_RMW from the Fama–French five-factor model) and Monte Carlo–driven Sharpe-ratio optimization. Monthly option rolls then overlay:

1. **Covered Calls** — harvesting option premiums to generate yield and provide modest downside buffer in calm markets.  
2. **Dynamic Collar** — funding protective puts with call premiums and scaling both strikes by the VIX index to enforce a hard floor on losses during volatile regimes.  

Backtests span January 2016 through December 2024, with performance benchmarked against the S&P 500 (SPY). Results include growth-of-$1 curves, drawdown analyses, rolling Sharpe ratios, VaR/CVaR comparisons, and monthly return heatmaps for each strategy :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}.

---

## Key Features

- **Factor Screening & Portfolio Optimization**  
  - β_RMW signal computed via rolling OLS regressions on Fama–French factors, smoothed with a 21-day EMA.  
  - Monte Carlo simulation (10,000 portfolios per rebalance) to maximize the Sharpe ratio under long-only constraints.  

- **Option Pricing Engine**  
  - American-style binomial (Cox–Ross–Rubinstein) tree incorporating early exercise, discrete dividends, and a volatility skew uplift for puts.  
  - Realistic trading friction adjustments: bid/ask spread, slippage, and commission rates.  

- **VIX-Driven Strike Selection**  
  - Strikes scale linearly between 5 %–10 % OTM based on the VIX level at each 30-day roll.  

- **Cash Account Accounting**  
  - Net premiums and option payoffs accumulate in a separate cash leg, combined with equity value to form “covered-value” and “collar-value” series.  

- **Rich Analytics & Visualizations**  
  - Growth-of-$1 charts for equity, covered calls, collar, and SPY.  
  - Drawdown comparisons and 12-month rolling Sharpe plots.  
  - VaR₉₅ and CVaR₉₅ bar charts.  
  - Monthly return heatmaps with annotated rebalance dates.  

---

## Repository Structure

## Overview

python -m scripts.database_initialization
python -m scripts.run_db_pipeline

How to run:
## Overview

How to run:
# 1. Fundamentals preprocessing
python -m scripts.fundamentals           --start 2016-01-10 --end 2025-04-11

# 2. Hybrid‑portfolio construction (beta‑RMW + Monte‑Carlo)
python -m scripts.beta_rmw_fundamentals  --output_dir ./results --high_quantile 0.60 --low_quantile 0.00 --rebalance_frequency 21 --roll_lookback 21

# 3. Pure covered‑calls back‑test
python -m scripts.covered_calls          --start 2016-01-10 --end 2025-04-11 --capital 100000000 --otm_delta 0.05 --expiry_days 30 --risk_free_rate 0.02 --volatility 0.25 --dividend_yield 0.02 --commission_rate 0.001 --vol_lookback 30 --use_dynamic_vol 1 --steps 200

# 4. Combined equity + covered‑calls strategy
python -m scripts.combined_strategy_rmw_calls --start 2016-01-10 --end 2025-04-11 --capital 100000000 --otm_delta 0.05 --expiry_days 30 --risk_free_rate 0.02 --volatility 0.25 --dividend_yield 0.02 --commission_rate 0.001 --vol_lookback 30 --use_dynamic_vol --steps 200 --equity_weight 0.7

# 5. Collar Hedging 
python -m scripts.collar_hedging --start 2016-01-01 --end 2024-12-31 --floor 0.97 --cap 1.05 --expiry_days 30 --vol_lookback 21 --r 0.02 --q 0.0 --commission 0.001 --steps 200 --skew 0.10 --div_dates 2020-06-15,2021-06-15 --div_amounts 0.5,0.5 --spread_pct 0.005 --slippage_pct 0.001 --initial-capital 100000000

# 6. Generate visual report
python -m scripts.visuals


This project provides a comprehensive framework to perform portfolio optimization using the Efficient Frontier strategy specifically tailored for Nasdaq-100 stocks. It integrates fundamental and technical analysis, Monte Carlo simulations, and CAPM/Fama-French factor models to maximize returns while managing risk effectively.

## Features

- **Data Management:** Automated fetching and storage of Nasdaq-100 tickers and historical price data.
- **Fundamental Analysis:** Calculation of expected returns, beta values, and factor analysis using CAPM and Fama-French models.
- **Technical Indicators:** Computation of critical signals including SMA, RSI, MACD, and ATR.
- **Portfolio Optimization:** Utilization of Monte Carlo simulations to generate optimal portfolio weights that maximize the Sharpe Ratio.
- **Backtesting:** Robust testing capabilities to evaluate the historical performance of various investment strategies.

## Directory Structure

```
buff-quants-nq_efficientfrontier_strategy/
├── README.md
├── config.py
├── LICENSE
├── requirements.txt
├── database/
│   ├── data.db
│   └── schema.sql
├── docs/
│   └── NQ_Efficient_Frontier_Project_Outline.docx
├── logs/
├── results/
│   ├── [various fundamental analysis CSV files]
└── scripts/
    ├── backtest.py
    ├── cleanup_database.py
    ├── db_setup.py
    ├── fetch_price.py
    ├── fetch_tickers.py
    ├── fundamentals.py
    ├── monte_carlo.py
    ├── optimization.py
    ├── run_analysis.py
    └── technical_signals.py
```

## Installation

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Results Folder

- Portfolio weights and performance metrics stored in the SQLite database.
- Logs and data persistence for analysis reproducibility.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Authors

Buff Quants - 2025

