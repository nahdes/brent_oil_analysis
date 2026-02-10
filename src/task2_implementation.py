"""
Task 2: Bayesian Change Point Detection - Python Script Implementation
Brent Oil Price Analysis

This script implements Bayesian change point detection using PyMC to identify
structural breaks in Brent oil prices.

Author: Birhan Energies Data Science Team
Date: February 2026

Usage:
    python task2_implementation.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# PyMC imports
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    print("❌ PyMC not installed!")
    print("   Install with: pip install pymc arviz")
    exit(1)

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent if Path(__file__).parent.name == 'src' else Path(__file__).parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_EVENTS = PROJECT_ROOT / 'data' / 'events'
RESULTS_FIGURES = PROJECT_ROOT / 'results' / 'figures'
RESULTS_MODELS = PROJECT_ROOT / 'results' / 'models'

# Create directories
RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
RESULTS_MODELS.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("TASK 2: BAYESIAN CHANGE POINT DETECTION")
print("Brent Oil Price Analysis")
print("=" * 80)
print(f"\nPyMC version: {pm.__version__}")
print(f"ArviZ version: {az.__version__}")
print(f"\nProject root: {PROJECT_ROOT}")
print(f"Results directory: {RESULTS_FIGURES.parent}\n")


def load_data():
    """Load and prepare data for analysis."""
    print("=" * 80)
    print("STEP 1: DATA LOADING")
    print("=" * 80)
    
    # Load prices
    df_prices = pd.read_csv(
        DATA_RAW / 'BrentOilPrices.csv',
        parse_dates=['Date'],
        dayfirst=True
    )
    df_prices = df_prices.sort_values('Date').reset_index(drop=True)
    
    # Load events
    df_events = pd.read_csv(
        DATA_EVENTS / 'Major_Oil_Events_Enhanced.csv',
        parse_dates=['Date']
    )
    
    print(f"\n✓ Loaded {len(df_prices):,} daily price observations")
    print(f"  Date range: {df_prices['Date'].min().date()} to {df_prices['Date'].max().date()}")
    print(f"\n✓ Loaded {len(df_events)} major events")
    
    return df_prices, df_events


def prepare_analysis_data(df_prices, start_date='2014-01-01', end_date='2016-12-31'):
    """Prepare data for specific analysis period."""
    print("\n" + "=" * 80)
    print("STEP 2: DATA PREPARATION")
    print("=" * 80)
    
    # Calculate log returns
    df_prices['Log_Returns'] = np.log(df_prices['Price']) - np.log(df_prices['Price'].shift(1))
    df_prices = df_prices.dropna().reset_index(drop=True)
    
    # Filter to analysis period
    df_analysis = df_prices[
        (df_prices['Date'] >= start_date) & 
        (df_prices['Date'] <= end_date)
    ].copy()
    df_analysis = df_analysis.reset_index(drop=True)
    
    print(f"\n✓ Analysis period: {df_analysis['Date'].min().date()} to {df_analysis['Date'].max().date()}")
    print(f"  Observations: {len(df_analysis):,}")
    print(f"  Mean log return: {df_analysis['Log_Returns'].mean():.6f}")
    print(f"  Std log return: {df_analysis['Log_Returns'].std():.6f}")
    
    # Visualize
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Raw prices
    axes[0].plot(df_analysis['Date'], df_analysis['Price'], linewidth=1, color='darkblue')
    axes[0].set_title(f'Brent Oil Prices ({start_date} to {end_date})', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Price (USD/barrel)')
    axes[0].grid(True, alpha=0.3)
    
    # Log returns
    axes[1].plot(df_analysis['Date'], df_analysis['Log_Returns'], linewidth=0.5, color='darkgreen', alpha=0.6)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1].set_title('Daily Log Returns', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Log Returns')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_FIGURES / 'task2_data_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Data visualization saved to: {RESULTS_FIGURES / 'task2_data_overview.png'}")
    
    return df_analysis


def build_and_sample_model(df_analysis):
    """Build and sample from Bayesian change point model."""
    print("\n" + "=" * 80)
    print("STEP 3: BAYESIAN MODEL BUILDING & SAMPLING")
    print("=" * 80)
    
    n = len(df_analysis)
    log_returns = df_analysis['Log_Returns'].values
    
    print(f"\nBuilding change point model with {n} observations...")
    
    # Build model
    with pm.Model() as model:
        # Prior for change point location
        tau = pm.DiscreteUniform('tau', lower=0, upper=n-1)
        
        # Priors for means before/after
        mu_1 = pm.Normal('mu_before', mu=log_returns.mean(), sigma=log_returns.std()*2)
        mu_2 = pm.Normal('mu_after', mu=log_returns.mean(), sigma=log_returns.std()*2)
        
        # Prior for standard deviation
        sigma = pm.HalfNormal('sigma', sigma=log_returns.std()*2)
        
        # Switch function
        idx = np.arange(n)
        mu = pm.math.switch(tau >= idx, mu_1, mu_2)
        
        # Likelihood
        obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=log_returns)
    
    print("✓ Model defined")
    print("\nModel structure:")
    print(model)
    
    # Sample
    print("\n" + "-" * 80)
    print("Running MCMC sampling...")
    print("This may take 5-15 minutes depending on your system.\n")
    
    with model:
        trace = pm.sample(
            draws=2000,
            tune=1000,
            chains=4,
            target_accept=0.95,
            random_seed=RANDOM_SEED,
            return_inferencedata=True
        )
    
    print("\n✓ Sampling complete!")
    
    return trace, model


def check_convergence(trace):
    """Check MCMC convergence diagnostics."""
    print("\n" + "=" * 80)
    print("STEP 4: CONVERGENCE DIAGNOSTICS")
    print("=" * 80)
    
    # Summary statistics
    summary = az.summary(trace, var_names=['tau', 'mu_before', 'mu_after', 'sigma'])
    
    print("\nPosterior Summary:")
    print(summary)
    
    # Check diagnostics
    all_converged = (summary['r_hat'] < 1.01).all()
    
    print("\n" + "-" * 80)
    print("Convergence Check:")
    print(f"  All r_hat < 1.01: {'✓ YES' if all_converged else '✗ NO'}")
    print(f"  ESS bulk > 400: {'✓ YES' if (summary['ess_bulk'] > 400).all() else '✗ NO'}")
    
    if not all_converged:
        print("\n⚠ WARNING: Model may not have converged!")
        print("  Consider: increasing draws, tuning, or target_accept")
    
    # Plot trace
    axes = az.plot_trace(
        trace,
        var_names=['tau', 'mu_before', 'mu_after', 'sigma'],
        figsize=(14, 10)
    )
    plt.suptitle('MCMC Trace Plots', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_FIGURES / 'task2_trace_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Trace plots saved to: {RESULTS_FIGURES / 'task2_trace_plots.png'}")
    
    return summary


def identify_changepoint(trace, df_analysis):
    """Identify and visualize the change point."""
    print("\n" + "=" * 80)
    print("STEP 5: CHANGE POINT IDENTIFICATION")
    print("=" * 80)
    
    # Extract tau samples
    tau_samples = trace.posterior['tau'].values.flatten()
    
    # Get mode and mean
    tau_mode = int(stats.mode(tau_samples, keepdims=False)[0])
    tau_mean = int(tau_samples.mean())
    
    # Convert to date
    changepoint_date = df_analysis.loc[tau_mode, 'Date']
    
    # Calculate credible interval
    tau_ci = np.percentile(tau_samples, [2.5, 97.5])
    
    print(f"\nMost Probable Change Point:")
    print(f"  Index: {tau_mode}")
    print(f"  Date: {changepoint_date.date()}")
    print(f"  Mean index: {tau_mean}")
    print(f"  95% HDI: [{int(tau_ci[0])}, {int(tau_ci[1])}]")
    
    # Plot posterior of tau
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(tau_samples, bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=True)
    ax.axvline(tau_mode, color='red', linestyle='--', linewidth=2, 
               label=f'Mode: {changepoint_date.date()}')
    ax.axvline(tau_mean, color='orange', linestyle='--', linewidth=2, label=f'Mean: {tau_mean}')
    ax.set_xlabel('Time Index', fontsize=12)
    ax.set_ylabel('Posterior Probability Density', fontsize=12)
    ax.set_title('Posterior Distribution of Change Point (τ)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_FIGURES / 'task2_tau_posterior.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Change point posterior saved to: {RESULTS_FIGURES / 'task2_tau_posterior.png'}")
    
    return tau_mode, changepoint_date


def quantify_impact(trace, df_analysis, tau_mode):
    """Quantify the impact of the change point."""
    print("\n" + "=" * 80)
    print("STEP 6: IMPACT QUANTIFICATION")
    print("=" * 80)
    
    # Extract posterior samples
    mu_before_samples = trace.posterior['mu_before'].values.flatten()
    mu_after_samples = trace.posterior['mu_after'].values.flatten()
    
    # Calculate statistics
    mu_before_mean = mu_before_samples.mean()
    mu_after_mean = mu_after_samples.mean()
    mu_diff = mu_after_mean - mu_before_mean
    
    # Price levels
    price_before = df_analysis.loc[:tau_mode, 'Price'].mean()
    price_after = df_analysis.loc[tau_mode:, 'Price'].mean()
    price_change_pct = ((price_after - price_before) / price_before) * 100
    
    # Credible intervals
    mu_before_ci = np.percentile(mu_before_samples, [2.5, 97.5])
    mu_after_ci = np.percentile(mu_after_samples, [2.5, 97.5])
    
    # Probabilities
    prob_decrease = (mu_after_samples < mu_before_samples).mean()
    prob_increase = (mu_after_samples > mu_before_samples).mean()
    
    print("\nImpact Analysis:")
    print("-" * 80)
    print(f"\nLog Returns:")
    print(f"  Mean before: {mu_before_mean:.6f} [{mu_before_ci[0]:.6f}, {mu_before_ci[1]:.6f}]")
    print(f"  Mean after:  {mu_after_mean:.6f} [{mu_after_ci[0]:.6f}, {mu_after_ci[1]:.6f}]")
    print(f"  Difference:  {mu_diff:.6f}")
    
    print(f"\nPrice Levels:")
    print(f"  Average before: ${price_before:.2f}")
    print(f"  Average after:  ${price_after:.2f}")
    print(f"  Change:         {price_change_pct:+.2f}%")
    
    print(f"\nProbabilistic Statements:")
    print(f"  P(decrease): {prob_decrease*100:.1f}%")
    print(f"  P(increase): {prob_increase*100:.1f}%")
    
    # Visualize posteriors
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Before
    axes[0].hist(mu_before_samples, bins=50, alpha=0.7, color='darkblue', 
                 edgecolor='black', density=True)
    axes[0].axvline(mu_before_mean, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {mu_before_mean:.6f}')
    axes[0].set_xlabel('Mean Log Return')
    axes[0].set_ylabel('Posterior Density')
    axes[0].set_title('Before Change Point (μ₁)', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # After
    axes[1].hist(mu_after_samples, bins=50, alpha=0.7, color='darkgreen',
                 edgecolor='black', density=True)
    axes[1].axvline(mu_after_mean, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {mu_after_mean:.6f}')
    axes[1].set_xlabel('Mean Log Return')
    axes[1].set_ylabel('Posterior Density')
    axes[1].set_title('After Change Point (μ₂)', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_FIGURES / 'task2_mu_posteriors.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Posterior distributions saved to: {RESULTS_FIGURES / 'task2_mu_posteriors.png'}")
    
    return {
        'mu_before_mean': mu_before_mean,
        'mu_after_mean': mu_after_mean,
        'price_before': price_before,
        'price_after': price_after,
        'price_change_pct': price_change_pct,
        'prob_decrease': prob_decrease
    }


def visualize_changepoint(df_analysis, changepoint_date, tau_mode, impact):
    """Visualize the change point on price series."""
    print("\n" + "=" * 80)
    print("STEP 7: VISUALIZATION")
    print("=" * 80)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot prices
    ax.plot(df_analysis['Date'], df_analysis['Price'], linewidth=1.5, 
            color='darkblue', label='Brent Oil Price')
    
    # Mark change point
    ax.axvline(changepoint_date, color='red', linestyle='--', linewidth=2,
               label=f'Change Point: {changepoint_date.date()}')
    
    # Shade regions
    ax.axvspan(df_analysis['Date'].min(), changepoint_date, alpha=0.1, color='blue')
    ax.axvspan(changepoint_date, df_analysis['Date'].max(), alpha=0.1, color='green')
    
    # Average prices
    ax.axhline(impact['price_before'], color='blue', linestyle=':', linewidth=1.5,
               label=f"Avg Before: ${impact['price_before']:.2f}")
    ax.axhline(impact['price_after'], color='green', linestyle=':', linewidth=1.5,
               label=f"Avg After: ${impact['price_after']:.2f}")
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD/barrel)', fontsize=12)
    ax.set_title('Detected Change Point in Brent Oil Prices', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_FIGURES / 'task2_changepoint_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Change point visualization saved to: {RESULTS_FIGURES / 'task2_changepoint_visualization.png'}")


def associate_events(df_events, changepoint_date, impact):
    """Associate change point with major events."""
    print("\n" + "=" * 80)
    print("STEP 8: EVENT ASSOCIATION")
    print("=" * 80)
    
    window_days = 30
    
    nearby_events = df_events[
        (df_events['Date'] >= changepoint_date - pd.Timedelta(days=window_days)) &
        (df_events['Date'] <= changepoint_date + pd.Timedelta(days=window_days))
    ]
    
    print(f"\nEvents within ±{window_days} days of {changepoint_date.date()}:")
    print("=" * 80)
    
    if len(nearby_events) > 0:
        for idx, event in nearby_events.iterrows():
            days_diff = (event['Date'] - changepoint_date).days
            print(f"\n{event['Date'].date()} ({days_diff:+d} days)")
            print(f"  Type: {event['Event_Type']}")
            print(f"  {event['Description']}")
        
        # Generate conclusion
        event = nearby_events.iloc[0]
        print("\n" + "=" * 80)
        print("📊 CONCLUSION:")
        print("=" * 80)
        print(f"\nFollowing the {event['Event_Type']} around {event['Date'].date()},")
        print(f"the Bayesian model detects a structural break on {changepoint_date.date()}.")
        print(f"The average price shifted from ${impact['price_before']:.2f} to ${impact['price_after']:.2f},")
        print(f"a change of {impact['price_change_pct']:+.2f}%.")
        print(f"There is a {max(impact['prob_decrease'], 1-impact['prob_decrease'])*100:.1f}% probability")
        print(f"that this represents a genuine structural break.")
    else:
        print(f"\nNo events found within ±{window_days} days.")
        print(f"The change point on {changepoint_date.date()} may be driven by")
        print(f"market dynamics or events not in our dataset.")


def save_results(trace, summary, changepoint_date, tau_mode, impact):
    """Save all results to files."""
    print("\n" + "=" * 80)
    print("STEP 9: SAVING RESULTS")
    print("=" * 80)
    
    # Save trace
    trace.to_netcdf(RESULTS_MODELS / 'changepoint_trace.nc')
    
    # Save summary
    summary.to_csv(RESULTS_MODELS / 'changepoint_summary.csv')
    
    # Save results
    results_df = pd.DataFrame({
        'Metric': [
            'Change Point Date',
            'Change Point Index',
            'Mean Before',
            'Mean After',
            'Price Before (avg)',
            'Price After (avg)',
            'Price Change %',
            'Probability of Decrease'
        ],
        'Value': [
            changepoint_date.date(),
            tau_mode,
            f"{impact['mu_before_mean']:.6f}",
            f"{impact['mu_after_mean']:.6f}",
            f"${impact['price_before']:.2f}",
            f"${impact['price_after']:.2f}",
            f"{impact['price_change_pct']:+.2f}%",
            f"{impact['prob_decrease']*100:.1f}%"
        ]
    })
    
    results_df.to_csv(RESULTS_MODELS / 'changepoint_results.csv', index=False)
    
    print(f"\n✓ Results saved:")
    print(f"  - {RESULTS_MODELS / 'changepoint_trace.nc'}")
    print(f"  - {RESULTS_MODELS / 'changepoint_summary.csv'}")
    print(f"  - {RESULTS_MODELS / 'changepoint_results.csv'}")
    
    print(f"\n✓ Figures saved:")
    print(f"  - {RESULTS_FIGURES / 'task2_data_overview.png'}")
    print(f"  - {RESULTS_FIGURES / 'task2_trace_plots.png'}")
    print(f"  - {RESULTS_FIGURES / 'task2_tau_posterior.png'}")
    print(f"  - {RESULTS_FIGURES / 'task2_mu_posteriors.png'}")
    print(f"  - {RESULTS_FIGURES / 'task2_changepoint_visualization.png'}")


def main():
    """Main execution function."""
    # Load data
    df_prices, df_events = load_data()
    
    # Prepare analysis data
    df_analysis = prepare_analysis_data(df_prices, start_date='2014-01-01', end_date='2016-12-31')
    
    # Build and sample model
    trace, model = build_and_sample_model(df_analysis)
    
    # Check convergence
    summary = check_convergence(trace)
    
    # Identify change point
    tau_mode, changepoint_date = identify_changepoint(trace, df_analysis)
    
    # Quantify impact
    impact = quantify_impact(trace, df_analysis, tau_mode)
    
    # Visualize
    visualize_changepoint(df_analysis, changepoint_date, tau_mode, impact)
    
    # Associate with events
    associate_events(df_events, changepoint_date, impact)
    
    # Save results
    save_results(trace, summary, changepoint_date, tau_mode, impact)
    
    print("\n" + "=" * 80)
    print("✅ TASK 2 ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {RESULTS_FIGURES.parent}")
    print("\nNext steps:")
    print("  1. Review visualizations in results/figures/")
    print("  2. Check model diagnostics (R-hat < 1.01)")
    print("  3. Document findings")
    print("  4. Proceed to Task 3: Dashboard Development")
    print("\n")


if __name__ == "__main__":
    main()
