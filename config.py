"""
Configuration file for Brent Oil Analysis project
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EVENTS_DATA_DIR = DATA_DIR / "events"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
TABLES_DIR = RESULTS_DIR / "tables"
REPORTS_DIR = RESULTS_DIR / "reports"

# Logging
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOGS_DIR / "analysis.log"

# Data file paths
BRENT_PRICES_FILE = RAW_DATA_DIR / "BrentOilPrices.csv"
EVENTS_FILE = EVENTS_DATA_DIR / "Major_Oil_Events.csv"

# Analysis parameters
RANDOM_SEED = 42
CONFIDENCE_LEVEL = 0.95
EVENT_WINDOW_DAYS = 30  # Days before/after event to analyze

# Visualization settings
FIGURE_DPI = 300
FIGURE_FORMAT = "png"

# Model parameters
MCMC_SAMPLES = 2000
MCMC_TUNE = 1000
MCMC_CHAINS = 4
TARGET_ACCEPT = 0.9

# Dashboard settings
DASHBOARD_HOST = "localhost"
DASHBOARD_PORT = 5000
DASHBOARD_DEBUG = True

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EVENTS_DATA_DIR,
                  RESULTS_DIR, FIGURES_DIR, MODELS_DIR, TABLES_DIR, REPORTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
