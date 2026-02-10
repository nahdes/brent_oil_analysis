"""
Task 3: Flask Backend API
Brent Oil Price Analysis Dashboard

This Flask application serves data and analysis results for the interactive dashboard.

Author: Birhan Energies Data Science Team
Date: February 2026
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

# Initialize Flask app
app = Flask(__name__, static_folder=Path(__file__).parent.parent / 'frontend' / 'build', static_url_path='/')
CORS(app)  # Enable CORS for React frontend

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_EVENTS = PROJECT_ROOT / 'data' / 'events'
RESULTS_MODELS = PROJECT_ROOT / 'results' / 'models'
RESULTS_TABLES = PROJECT_ROOT / 'results' / 'tables'

# Global data cache
data_cache = {}


@app.route('/')
def root():
    """Serve the root index page or API info."""
    try:
        return app.send_static_file('index.html')
    except:
        # If React build doesn't exist, return API info
        return jsonify({
            'message': 'Brent Oil Analysis Dashboard API',
            'version': '1.0.0',
            'docs': 'Available endpoints listed below',
            'endpoints': {
                'health': '/api/health',
                'prices': '/api/prices',
                'events': '/api/events',
                'analysis': '/api/analysis/*'
            }
        })


@app.route('/favicon.ico')
def favicon():
    """Serve favicon."""
    return app.send_static_file('favicon.ico'), 200


def load_data():
    """Load all data into memory for faster access."""
    global data_cache
    
    if data_cache:
        return data_cache
    
    print("Loading data into cache...")
    
    # Load price data
    df_prices = pd.read_csv(
        DATA_RAW / 'BrentOilPrices.csv',
        parse_dates=['Date']
    )
    df_prices = df_prices.sort_values('Date').reset_index(drop=True)
    
    # Calculate additional metrics
    df_prices['Returns'] = df_prices['Price'].pct_change()
    df_prices['Log_Returns'] = np.log(df_prices['Price']) - np.log(df_prices['Price'].shift(1))
    df_prices['MA_30'] = df_prices['Price'].rolling(window=30).mean()
    df_prices['MA_90'] = df_prices['Price'].rolling(window=90).mean()
    df_prices['Volatility_30'] = df_prices['Returns'].rolling(window=30).std() * np.sqrt(252)
    
    # Load events
    df_events = pd.read_csv(
        DATA_EVENTS / 'Major_Oil_Events_Enhanced.csv',
        parse_dates=['Date']
    )
    
    # Load event impacts if available
    event_impacts_file = RESULTS_TABLES / 'event_impact_analysis.csv'
    if event_impacts_file.exists():
        df_event_impacts = pd.read_csv(event_impacts_file)
    else:
        df_event_impacts = None
    
    # Load change point results if available
    changepoint_file = RESULTS_MODELS / 'changepoint_results.csv'
    if changepoint_file.exists():
        df_changepoints = pd.read_csv(changepoint_file)
    else:
        df_changepoints = None
    
    data_cache = {
        'prices': df_prices,
        'events': df_events,
        'event_impacts': df_event_impacts,
        'changepoints': df_changepoints
    }
    
    print(f"✓ Loaded {len(df_prices)} price records")
    print(f"✓ Loaded {len(df_events)} events")
    
    return data_cache


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


@app.route('/api/prices', methods=['GET'])
def get_prices():
    """
    Get historical price data with optional filtering.
    
    Query Parameters:
    - start_date: YYYY-MM-DD (optional)
    - end_date: YYYY-MM-DD (optional)
    - include_indicators: true/false (default: false)
    """
    data = load_data()
    df = data['prices'].copy()
    
    # Filter by date range
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    if start_date:
        df = df[df['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['Date'] <= pd.to_datetime(end_date)]
    
    # Include indicators
    include_indicators = request.args.get('include_indicators', 'false').lower() == 'true'
    
    if include_indicators:
        columns = ['Date', 'Price', 'Returns', 'Log_Returns', 'MA_30', 'MA_90', 'Volatility_30']
    else:
        columns = ['Date', 'Price']
    
    # Convert to JSON-friendly format
    result = df[columns].copy()
    result['Date'] = result['Date'].dt.strftime('%Y-%m-%d')
    
    return jsonify({
        'data': result.to_dict('records'),
        'count': len(result),
        'start_date': result['Date'].iloc[0] if len(result) > 0 else None,
        'end_date': result['Date'].iloc[-1] if len(result) > 0 else None
    })


@app.route('/api/prices/summary', methods=['GET'])
def get_price_summary():
    """Get summary statistics for price data."""
    data = load_data()
    df = data['prices']
    
    # Calculate statistics
    summary = {
        'count': int(len(df)),
        'mean': float(df['Price'].mean()),
        'median': float(df['Price'].median()),
        'std': float(df['Price'].std()),
        'min': float(df['Price'].min()),
        'max': float(df['Price'].max()),
        'current': float(df['Price'].iloc[-1]),
        'date_range': {
            'start': df['Date'].min().strftime('%Y-%m-%d'),
            'end': df['Date'].max().strftime('%Y-%m-%d')
        },
        'volatility': {
            'current': float(df['Volatility_30'].iloc[-1]) if pd.notna(df['Volatility_30'].iloc[-1]) else None,
            'mean': float(df['Volatility_30'].mean()),
            'max': float(df['Volatility_30'].max())
        }
    }
    
    return jsonify(summary)


@app.route('/api/events', methods=['GET'])
def get_events():
    """
    Get major events data.
    
    Query Parameters:
    - event_type: Filter by event type (optional)
    - start_date: YYYY-MM-DD (optional)
    - end_date: YYYY-MM-DD (optional)
    """
    data = load_data()
    df = data['events'].copy()
    
    # Filter by event type
    event_type = request.args.get('event_type')
    if event_type:
        df = df[df['Event_Type'] == event_type]
    
    # Filter by date range
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    if start_date:
        df = df[df['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['Date'] <= pd.to_datetime(end_date)]
    
    # Convert to JSON-friendly format
    result = df.copy()
    result['Date'] = result['Date'].dt.strftime('%Y-%m-%d')
    
    return jsonify({
        'data': result.to_dict('records'),
        'count': len(result)
    })


@app.route('/api/events/types', methods=['GET'])
def get_event_types():
    """Get unique event types."""
    data = load_data()
    df = data['events']
    
    event_types = df['Event_Type'].unique().tolist()
    
    # Count by type
    type_counts = df['Event_Type'].value_counts().to_dict()
    
    return jsonify({
        'types': event_types,
        'counts': type_counts
    })


@app.route('/api/events/<event_id>/impact', methods=['GET'])
def get_event_impact(event_id):
    """Get price impact for a specific event."""
    data = load_data()
    df_events = data['events']
    df_prices = data['prices']
    
    try:
        event_idx = int(event_id)
        event = df_events.iloc[event_idx]
    except:
        return jsonify({'error': 'Event not found'}), 404
    
    event_date = pd.to_datetime(event['Date'])
    
    # Calculate impact (30 days before/after)
    window = 30
    
    # Find event in price data
    event_price_idx = (df_prices['Date'] - event_date).abs().idxmin()
    
    before_idx = max(0, event_price_idx - window)
    after_idx = min(len(df_prices) - 1, event_price_idx + window)
    
    price_before = df_prices.loc[before_idx, 'Price']
    price_event = df_prices.loc[event_price_idx, 'Price']
    price_after = df_prices.loc[after_idx, 'Price']
    
    change_pct = ((price_after - price_before) / price_before) * 100
    
    # Get price series around event
    start_idx = max(0, event_price_idx - 60)
    end_idx = min(len(df_prices) - 1, event_price_idx + 60)
    
    price_series = df_prices.loc[start_idx:end_idx, ['Date', 'Price']].copy()
    price_series['Date'] = price_series['Date'].dt.strftime('%Y-%m-%d')
    
    return jsonify({
        'event': {
            'date': event['Date'].strftime('%Y-%m-%d'),
            'type': event['Event_Type'],
            'description': event['Description']
        },
        'impact': {
            'price_before': float(price_before),
            'price_event': float(price_event),
            'price_after': float(price_after),
            'change_percent': float(change_pct),
            'window_days': window
        },
        'price_series': price_series.to_dict('records')
    })


@app.route('/api/analysis/volatility', methods=['GET'])
def get_volatility_analysis():
    """Get volatility analysis over time."""
    data = load_data()
    df = data['prices'].copy()
    
    # Calculate volatility by year
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    volatility_by_year = df.groupby('Year')['Volatility_30'].mean().dropna()
    
    # Calculate volatility by decade
    df['Decade'] = (df['Year'] // 10) * 10
    volatility_by_decade = df.groupby('Decade')['Volatility_30'].mean().dropna()
    
    return jsonify({
        'by_year': {
            'years': volatility_by_year.index.tolist(),
            'volatility': volatility_by_year.values.tolist()
        },
        'by_decade': {
            'decades': volatility_by_decade.index.tolist(),
            'volatility': volatility_by_decade.values.tolist()
        },
        'current': float(df['Volatility_30'].iloc[-1]) if pd.notna(df['Volatility_30'].iloc[-1]) else None
    })


@app.route('/api/analysis/changepoints', methods=['GET'])
def get_changepoints():
    """Get detected change points from Bayesian analysis."""
    data = load_data()
    
    if data['changepoints'] is None:
        return jsonify({
            'message': 'No change point analysis results available',
            'data': []
        })
    
    return jsonify({
        'data': data['changepoints'].to_dict('records')
    })


@app.route('/api/analysis/event-correlation', methods=['GET'])
def get_event_correlation():
    """Get correlation between events and price changes."""
    data = load_data()
    
    if data['event_impacts'] is None:
        # Calculate on the fly
        df_events = data['events']
        df_prices = data['prices']
        
        impacts = []
        for idx, event in df_events.iterrows():
            event_date = pd.to_datetime(event['Date'])
            event_price_idx = (df_prices['Date'] - event_date).abs().idxmin()
            
            before_idx = max(0, event_price_idx - 30)
            after_idx = min(len(df_prices) - 1, event_price_idx + 30)
            
            price_before = df_prices.loc[before_idx, 'Price']
            price_after = df_prices.loc[after_idx, 'Price']
            change_pct = ((price_after - price_before) / price_before) * 100
            
            impacts.append({
                'event_id': idx,
                'date': event['Date'].strftime('%Y-%m-%d'),
                'event_type': event['Event_Type'],
                'description': event['Description'][:60] + '...',
                'price_change': float(change_pct)
            })
        
        return jsonify({'data': impacts})
    else:
        result = data['event_impacts'].copy()
        if 'Date' in result.columns:
            result['Date'] = pd.to_datetime(result['Date']).dt.strftime('%Y-%m-%d')
        return jsonify({'data': result.to_dict('records')})


@app.route('/api/dashboard/stats', methods=['GET'])
def get_dashboard_stats():
    """Get key statistics for dashboard overview."""
    data = load_data()
    df_prices = data['prices']
    df_events = data['events']
    
    # Current price
    current_price = float(df_prices['Price'].iloc[-1])
    
    # Price change (30 days)
    price_30d_ago = float(df_prices['Price'].iloc[-30])
    change_30d = ((current_price - price_30d_ago) / price_30d_ago) * 100
    
    # Price change (1 year)
    price_1y_ago = float(df_prices['Price'].iloc[-252])
    change_1y = ((current_price - price_1y_ago) / price_1y_ago) * 100
    
    # All-time high and low
    all_time_high = float(df_prices['Price'].max())
    all_time_low = float(df_prices['Price'].min())
    
    # Current volatility
    current_volatility = float(df_prices['Volatility_30'].iloc[-1]) if pd.notna(df_prices['Volatility_30'].iloc[-1]) else 0
    
    # Recent events (last 90 days)
    recent_date = pd.Timestamp.now() - pd.Timedelta(days=90)
    recent_events = len(df_events[df_events['Date'] >= recent_date])
    
    return jsonify({
        'current_price': current_price,
        'change_30d': change_30d,
        'change_1y': change_1y,
        'all_time_high': all_time_high,
        'all_time_low': all_time_low,
        'current_volatility': current_volatility * 100,  # Convert to percentage
        'total_events': len(df_events),
        'recent_events': recent_events,
        'data_points': len(df_prices),
        'date_range': {
            'start': df_prices['Date'].min().strftime('%Y-%m-%d'),
            'end': df_prices['Date'].max().strftime('%Y-%m-%d')
        }
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("=" * 80)
    print("BRENT OIL ANALYSIS - FLASK BACKEND")
    print("Task 3: Interactive Dashboard API")
    print("=" * 80)
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_RAW}")
    
    # Load data on startup
    load_data()
    
    print("\n" + "=" * 80)
    print("Starting Flask server...")
    print("API will be available at: http://localhost:5000")
    print("=" * 80)
    print("\nAvailable endpoints:")
    print("  GET  /api/health")
    print("  GET  /api/prices")
    print("  GET  /api/prices/summary")
    print("  GET  /api/events")
    print("  GET  /api/events/types")
    print("  GET  /api/events/<id>/impact")
    print("  GET  /api/analysis/volatility")
    print("  GET  /api/analysis/changepoints")
    print("  GET  /api/analysis/event-correlation")
    print("  GET  /api/dashboard/stats")
    print("\n" + "=" * 80 + "\n")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)