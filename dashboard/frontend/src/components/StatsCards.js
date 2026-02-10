import React from 'react';
import './StatsCards.css';

const StatsCards = ({ stats }) => {
  if (!stats) return null;

  const formatNumber = (num) => {
    return new Intl.NumberFormat('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(num);
  };

  const formatPercent = (num) => {
    const sign = num >= 0 ? '+' : '';
    return `${sign}${formatNumber(num)}%`;
  };

  const getChangeClass = (value) => {
    if (value > 0) return 'positive';
    if (value < 0) return 'negative';
    return 'neutral';
  };

  return (
    <div className="stats-cards">
      <div className="stat-card">
        <div className="stat-icon">💰</div>
        <div className="stat-content">
          <h3>Current Price</h3>
          <p className="stat-value">${formatNumber(stats.current_price)}</p>
          <p className="stat-label">per barrel</p>
        </div>
      </div>

      <div className="stat-card">
        <div className="stat-icon">📈</div>
        <div className="stat-content">
          <h3>30-Day Change</h3>
          <p className={`stat-value ${getChangeClass(stats.change_30d)}`}>
            {formatPercent(stats.change_30d)}
          </p>
          <p className="stat-label">last month</p>
        </div>
      </div>

      <div className="stat-card">
        <div className="stat-icon">📊</div>
        <div className="stat-content">
          <h3>1-Year Change</h3>
          <p className={`stat-value ${getChangeClass(stats.change_1y)}`}>
            {formatPercent(stats.change_1y)}
          </p>
          <p className="stat-label">annual</p>
        </div>
      </div>

      <div className="stat-card">
        <div className="stat-icon">⚡</div>
        <div className="stat-content">
          <h3>Volatility</h3>
          <p className="stat-value">{formatNumber(stats.current_volatility)}%</p>
          <p className="stat-label">30-day annualized</p>
        </div>
      </div>

      <div className="stat-card">
        <div className="stat-icon">🔝</div>
        <div className="stat-content">
          <h3>All-Time High</h3>
          <p className="stat-value">${formatNumber(stats.all_time_high)}</p>
          <p className="stat-label">maximum</p>
        </div>
      </div>

      <div className="stat-card">
        <div className="stat-icon">📍</div>
        <div className="stat-content">
          <h3>Major Events</h3>
          <p className="stat-value">{stats.total_events}</p>
          <p className="stat-label">tracked events</p>
        </div>
      </div>
    </div>
  );
};

export default StatsCards;
