# Brent Oil Price Change Point Analysis

**10 Academy: Artificial Intelligence Mastery - Week 11 Challenge**

## Project Overview

This project analyzes how major political and economic events affect Brent oil prices using Bayesian change point detection. The analysis focuses on identifying structural breaks in oil prices and associating them with geopolitical events, OPEC policy changes, and economic shocks.

## Business Context

**Client**: Birhan Energies - Leading consultancy in the energy sector

**Objectives**:
- Identify key events that significantly impacted Brent oil prices
- Quantify event impacts using statistical methods
- Provide data-driven insights for investment strategies and policy development

## Project Structure

```
brent_oil_analysis/
├── data/                   # Data files and datasets
│   ├── raw/               # Original Brent oil prices
│   ├── processed/         # Cleaned data
│   ├── events/            # Major events dataset
│   └── external/          # Additional data sources
├── src/                   # Source code
│   ├── data_processing/   # Data loading and preprocessing
│   ├── models/            # Bayesian change point models
│   ├── visualization/     # Plotting utilities
│   └── utils/             # Helper functions
├── notebooks/             # Jupyter notebooks
│   ├── exploratory/       # EDA notebooks
│   ├── modeling/          # Modeling notebooks
│   └── analysis/          # Analysis notebooks
├── results/               # Analysis outputs
│   ├── figures/           # Visualizations
│   ├── models/            # Saved models
│   ├── tables/            # Summary tables
│   └── reports/           # Final reports
├── dashboard/             # Interactive dashboard
│   ├── backend/           # Flask API
│   └── frontend/          # React app
├── tests/                 # Unit tests
├── docs/                  # Documentation
└── logs/                  # Log files
```

## Installation

### Prerequisites

- Python 3.9+
- Node.js 16+ (for dashboard)
- Git

### Python Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dashboard Setup

```bash
cd dashboard

# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
```

## Usage

### Task 1: Foundation Analysis

```bash
python src/task1_analysis.py
```

### Task 2: Bayesian Change Point Modeling

```bash
jupyter notebook notebooks/modeling/task2_bayesian_changepoint.ipynb
```

### Task 3: Dashboard

```bash
# Start backend
cd dashboard/backend
python app.py

# Start frontend (in new terminal)
cd dashboard/frontend
npm start
```

## Data

- **Brent Oil Prices**: Daily prices from May 20, 1987 to November 14, 2022
- **Major Events**: 15 key geopolitical and economic events (1990-2022)

## Methodology

1. **Data Analysis Workflow**: Load, clean, and explore the data
2. **Time Series Properties**: Analyze stationarity, trend, and volatility
3. **Bayesian Change Point Detection**: Implement using PyMC
4. **Event Association**: Map detected change points to real-world events
5. **Impact Quantification**: Measure percentage shifts in prices

## Technologies

- **Statistical Analysis**: Python, Pandas, NumPy, SciPy
- **Bayesian Modeling**: PyMC, ArviZ
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Backend**: Flask, SQLAlchemy
- **Frontend**: React, Recharts, D3.js

## Team

- **Tutors**: Kerod, Filimon, Mahbubah
- **Communication**: Slack #all-week11

## Key Dates

- **Challenge Introduction**: Feb 04, 2026
- **Interim Submission**: Feb 08, 2026, 8:00 PM UTC
- **Final Submission**: Feb 10, 2026, 8:00 PM UTC

## Deliverables

### Task 1
- [x] Analysis workflow document
- [x] Events dataset (CSV)
- [x] Python implementation
- [x] EDA visualizations

### Task 2
- [ ] Bayesian change point model (PyMC)
- [ ] Posterior distributions
- [ ] Change point identification
- [ ] Event association analysis

### Task 3
- [ ] Flask backend API
- [ ] React frontend dashboard
- [ ] Interactive visualizations
- [ ] Deployment documentation

## License

This project is part of 10 Academy's AI Mastery program.

## References

- [Change Point Detection in Time Series](https://forecastegy.com/posts/change-point-detection-time-series-python/)
- [Bayesian Changepoint Detection with PyMC](https://www.pymc.io/)
- [Data Science Workflow](https://www.datascience-pm.com/data-science-workflow/)

---

**Birhan Energies** | Data-Driven Insights for the Energy Sector
