# Efficient Frontier Strategy for Nasdaq-100

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
python -m scripts.collar_hedging --start 2016-01-10 --end 2025-04-11 --equity_weight 0.7 --floor_scenarios 0.985 0.995 --cap_scenarios 1.03 1.05

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

## Usage

1. **Set Up the Database**

```bash
python scripts/db_setup.py
```

2. **Fetch Nasdaq-100 Tickers**

```bash
python scripts/fetch_tickers.py
```

2. **Fetch Historical Prices**

```bash
python scripts/fetch_price.py
```

3. **Run Fundamental and Technical Analysis**

```bash
python scripts/run_analysis.py
```

4. **Portfolio Optimization and Backtesting**

```bash
python scripts/optimization.py
python scripts/monte_carlo.py
python scripts/backtest.py
```

## Outputs

- Fundamental and technical analysis results in CSV format stored in `results/`
- Portfolio weights and performance metrics stored in the SQLite database and CSV files.
- Logs and data persistence for analysis reproducibility.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Authors

Buff Quants - 2025

