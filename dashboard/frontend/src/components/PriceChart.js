import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Scatter
} from 'recharts';

const PriceChart = ({ prices, events, highlightedEvent }) => {
  if (!prices || prices.length === 0) {
    return <div className="no-data">No price data available</div>;
  }

  // Prepare data for chart
  const chartData = prices.map(item => ({
    date: item.Date,
    price: parseFloat(item.Price),
    ma30: item.MA_30 ? parseFloat(item.MA_30) : null,
    ma90: item.MA_90 ? parseFloat(item.MA_90) : null
  }));

  // Custom tooltip
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="custom-tooltip">
          <p className="tooltip-date">{payload[0].payload.date}</p>
          <p className="tooltip-price">
            <strong>Price:</strong> ${payload[0].value.toFixed(2)}
          </p>
          {payload[1] && payload[1].value && (
            <p className="tooltip-ma">
              <strong>MA(30):</strong> ${payload[1].value.toFixed(2)}
            </p>
          )}
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
    <div className="price-chart-container">
      <ResponsiveContainer width="100%" height={400}>
        <LineChart
          data={chartData}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
          <XAxis 
            dataKey="date" 
            tickFormatter={formatXAxis}
            stroke="#666"
            style={{ fontSize: '12px' }}
          />
          <YAxis 
            label={{ value: 'Price (USD/barrel)', angle: -90, position: 'insideLeft' }}
            stroke="#666"
            style={{ fontSize: '12px' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          
          {/* Event markers */}
          {events && events.map((event, idx) => (
            <ReferenceLine
              key={idx}
              x={event.Date}
              stroke={highlightedEvent && highlightedEvent.Date === event.Date ? "#ff0000" : "#ff6b6b"}
              strokeWidth={highlightedEvent && highlightedEvent.Date === event.Date ? 2 : 1}
              strokeDasharray="3 3"
              label={{
                value: highlightedEvent && highlightedEvent.Date === event.Date ? '📍' : '',
                position: 'top'
              }}
            />
          ))}
          
          {/* Price line */}
          <Line
            type="monotone"
            dataKey="price"
            stroke="#2563eb"
            strokeWidth={2}
            dot={false}
            name="Brent Oil Price"
          />
          
          {/* Moving average lines */}
          <Line
            type="monotone"
            dataKey="ma30"
            stroke="#10b981"
            strokeWidth={1.5}
            dot={false}
            name="MA(30)"
            strokeDasharray="5 5"
          />
          <Line
            type="monotone"
            dataKey="ma90"
            stroke="#f59e0b"
            strokeWidth={1.5}
            dot={false}
            name="MA(90)"
            strokeDasharray="5 5"
          />
        </LineChart>
      </ResponsiveContainer>
      
      {highlightedEvent && (
        <div className="highlighted-event-info">
          <h4>📍 Highlighted Event</h4>
          <p><strong>{highlightedEvent.Date}</strong></p>
          <p>{highlightedEvent.Description}</p>
        </div>
      )}
    </div>
  );
};

export default PriceChart;
