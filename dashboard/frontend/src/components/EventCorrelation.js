import React, { useEffect, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  Legend
} from 'recharts';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const EventCorrelation = ({ events }) => {
  const [correlationData, setCorrelationData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchCorrelationData();
  }, []);

  const fetchCorrelationData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/analysis/event-correlation`);
      const result = await response.json();
      
      if (result.data && result.data.length > 0) {
        // Transform data for chart
        const chartData = result.data.map(item => ({
          name: item.event_type || item.Event_Type,
          impact: parseFloat(item.price_change || item['Change_%'] || 0),
          date: item.date || item.Date,
          description: item.description || item.Description
        }));
        setCorrelationData(chartData);
      }
      setLoading(false);
    } catch (err) {
      console.error('Error fetching correlation data:', err);
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="loading">Loading correlation data...</div>;
  }

  if (correlationData.length === 0) {
    return <div className="no-data">No correlation data available</div>;
  }

  // Custom tooltip
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="custom-tooltip">
          <p className="tooltip-title">{data.name}</p>
          {data.date && <p className="tooltip-date">{data.date}</p>}
          <p className={`tooltip-impact ${data.impact > 0 ? 'positive' : 'negative'}`}>
            <strong>Impact:</strong> {data.impact.toFixed(2)}%
          </p>
          {data.description && (
            <p className="tooltip-desc">{data.description}</p>
          )}
        </div>
      );
    }
    return null;
  };

  // Get color for bar based on impact
  const getBarColor = (value) => {
    if (value > 0) return '#10b981'; // Green for positive
    if (value < 0) return '#ef4444'; // Red for negative
    return '#6b7280'; // Gray for neutral
  };

  return (
    <div className="event-correlation-container">
      <ResponsiveContainer width="100%" height={350}>
        <BarChart
          data={correlationData}
          margin={{ top: 20, right: 30, left: 20, bottom: 80 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
          <XAxis
            dataKey="name"
            angle={-45}
            textAnchor="end"
            height={100}
            interval={0}
            style={{ fontSize: '11px' }}
          />
          <YAxis
            label={{
              value: 'Price Change (%)',
              angle: -90,
              position: 'insideLeft',
              style: { fontSize: '12px' }
            }}
            style={{ fontSize: '12px' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            content={() => (
              <div style={{ textAlign: 'center', marginTop: '10px', fontSize: '12px' }}>
                <span style={{ color: '#10b981', marginRight: '15px' }}>● Positive Impact</span>
                <span style={{ color: '#ef4444' }}>● Negative Impact</span>
              </div>
            )}
          />
          <Bar dataKey="impact" radius={[4, 4, 0, 0]}>
            {correlationData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getBarColor(entry.impact)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div className="correlation-summary">
        <p className="info-text">
          Price change within ±30 days of each event. Positive values indicate price increases, 
          negative values indicate decreases.
        </p>
      </div>
    </div>
  );
};

export default EventCorrelation;
