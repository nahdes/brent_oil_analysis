import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart
} from 'recharts';

const VolatilityChart = ({ prices }) => {
  if (!prices || prices.length === 0) {
    return <div className="no-data">No volatility data available</div>;
  }

  // Filter and prepare data
  const data = prices
    .filter(p => p.Volatility_30)
    .map(p => ({
      date: p.Date,
      volatility: parseFloat(p.Volatility_30) * 100, // Convert to percentage
      price: parseFloat(p.Price)
    }));

  // Custom tooltip
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="custom-tooltip">
          <p className="tooltip-date">{payload[0].payload.date}</p>
          <p className="tooltip-price">
            <strong>Volatility:</strong> {payload[0].value.toFixed(2)}%
          </p>
        </div>
      );
    }
    return null;
  };

  // Format date for X-axis
  const formatXAxis = (tickItem) => {
    const date = new Date(tickItem);
    return date.getFullYear();
  };

  return (
    <div className="volatility-chart-container">
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
          <XAxis 
            dataKey="date" 
            tickFormatter={formatXAxis}
            stroke="#666"
            style={{ fontSize: '12px' }}
          />
          <YAxis 
            label={{ 
              value: 'Volatility (%)', 
              angle: -90, 
              position: 'insideLeft',
              style: { fontSize: '12px' }
            }}
            stroke="#666"
            style={{ fontSize: '12px' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Area
            type="monotone"
            dataKey="volatility"
            stroke="#8b5cf6"
            fill="#8b5cf6"
            fillOpacity={0.3}
            strokeWidth={2}
          />
        </AreaChart>
      </ResponsiveContainer>

      <div className="volatility-info">
        <p className="info-text">
          30-day rolling volatility (annualized). Higher values indicate more price instability.
        </p>
      </div>
    </div>
  );
};

export default VolatilityChart;
