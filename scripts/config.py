"""
Configuration file for NQ_Efficient_Frontier project.

This file centralizes all constants and configurable parameters used throughout the project,
including database paths, technical indicator parameters, fundamental analysis constants,
optimization settings, and logging configurations.
"""


"""
Configuration file for Derivatives Final Project – Combined Portfolio Strategy

This file centralizes all constants and configurable parameters for:
  1. Data & Database Paths
  2. Fundamental Analysis (CAPM/Fama-French parameters)
  3. Portfolio Optimization (Beta_RMW screening, Monte Carlo settings)
  4. Covered Call Strategy (OTM delta, expiry settings, commission, etc.)
  5. Dynamic Collar Hedge (volatility/drawdown thresholds and adjustment parameters)
  6. Backtest Timing (start/end dates, rebalancing frequency)
  7. Logging & Debugging
"""

## API KEYS
# SEC for EDGAR Filings, Statements, and Financials
SEC_Base_URL = "https://api.sec-api.io"
SEC_API_KEY = "167b0f020738341948c35033b570748e599f5e632b62593d64e1c926967d28ac"

#Federal Reserve Economic Data (FRED) API
FRED_API_KEY = "e795295f1d454318e2ac436f480317d2"
FRED_Base_URL = "https://api.stlouisfed.org/fred"

############################
# PATHS & DATABASE
############################
DB_PATH = 'database/data.db'
LOG_DIR = 'logs'
RESULTS_DIR = 'results'

############################
# DATE & INVESTMENT PARAMETERS
############################
START_DATE = '2016-01-01'
END_DATE = '2025-04-15'
INIT_VALUE = 100000000        # Initial portfolio capital 100M
DATA_FREQ = '1D'

############################
# FUNDAMENTAL ANALYSIS PARAMETERS
############################
RISK_FREE_RATE = 0.02
MARKET_RISK_PREMIUM = 0.06
FUNDAMENTALS_MODEL = 'fama_french'  # Options: 'capm' or 'fama_french'
FUNDAMENTALS_START_DATE = '2020-01-01'

############################
# PORTFOLIO OPTIMIZATION SETTINGS
############################
# Beta_RMW screening thresholds
BETA_RMW_HIGH_QUANTILE = 0.60   # e.g. top 40% as eligible for selection
BETA_RMW_LOW_QUANTILE = 0.00    # e.g. no short positions by default
# Monte Carlo settings
NUM_SIMULATIONS = 10000
REBALANCE_FREQUENCY = 21        # Holding period in trading days for rebalancing
ROLL_LOOKBACK = 21            # For EMA smoothing of beta_RMW
PRICE_LOOKBACK = 126          # Days used to compute price covariance

############################
# COVERED CALL STRATEGY SETTINGS
############################
OTM_DELTA = 0.05              # Base delta for out-of-the-money calls
EXPIRY_DAYS = 30              # Days until option expiry
DIVIDEND_YIELD = 0.02
COMMISSION_RATE = 0.001
VOL_LOOKBACK = 30             # Lookback period for volatility estimation
USE_DYNAMIC_VOL = True        # Flag to use dynamic volatility adjustments
STEPS = 200                   # Steps for binomial tree simulation (if applicable)
EQUITY_WEIGHT = 1.0          # Weighting for equity in blended combined portfolio

############################
# COLLAR HEDGE SETTINGS (applied under high risk)
############################
# Trigger thresholds: set these based on either realized volatility or drawdown.
VOL_THRESHOLD = 0.22          # Annualized volatility threshold to trigger collars
DRAWDOWN_THRESHOLD = 0.10     # Trigger collar if drawdown exceeds 10%
# Collar adjustment parameters:
BASE_DELTA = 0.05             # Standard OTM delta
PROTECTIVE_DELTA = 0.10       # Delta used when in high risk conditions
FLOOR_MULTIPLIER = 0.985     # Minimum allowed multiplier (protective floor)
CAP_MULTIPLIER = 1.05         # Maximum allowed multiplier (call-sell cap)

############################
# LOGGING & DEBUGGING
############################
LOG_FILE = f"{LOG_DIR}/project.log"
LOG_LEVEL = 'INFO'
DEBUG_MODE = False
