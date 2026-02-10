import React, { useState, useEffect } from 'react';
import './App.css';
import PriceChart from './components/PriceChart';
import EventsPanel from './components/EventsPanel';
import StatsCards from './components/StatsCards';
import VolatilityChart from './components/VolatilityChart';
import EventCorrelation from './components/EventCorrelation';
import DateRangeFilter from './components/DateRangeFilter';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

function App() {
  const [prices, setPrices] = useState([]);
  const [events, setEvents] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [dateRange, setDateRange] = useState({
    start: '2010-01-01',
    end: new Date().toISOString().split('T')[0]
  });
  const [selectedEventType, setSelectedEventType] = useState('all');
  const [highlightedEvent, setHighlightedEvent] = useState(null);

  // Fetch dashboard stats
  useEffect(() => {
    fetchDashboardStats();
  }, []);

  // Fetch prices when date range changes
  useEffect(() => {
    fetchPrices();
  }, [dateRange]);

  // Fetch events when filters change
  useEffect(() => {
    fetchEvents();
  }, [dateRange, selectedEventType]);

  const fetchDashboardStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/dashboard/stats`);
      const data = await response.json();
      setStats(data);
    } catch (err) {
      console.error('Error fetching stats:', err);
      setError('Failed to load dashboard statistics');
    }
  };

  const fetchPrices = async () => {
    try {
      setLoading(true);
      const url = `${API_BASE_URL}/prices?start_date=${dateRange.start}&end_date=${dateRange.end}&include_indicators=true`;
      const response = await fetch(url);
      const result = await response.json();
      setPrices(result.data);
      setLoading(false);
    } catch (err) {
      console.error('Error fetching prices:', err);
      setError('Failed to load price data');
      setLoading(false);
    }
  };

  const fetchEvents = async () => {
    try {
      let url = `${API_BASE_URL}/events?start_date=${dateRange.start}&end_date=${dateRange.end}`;
      if (selectedEventType !== 'all') {
        url += `&event_type=${selectedEventType}`;
      }
      const response = await fetch(url);
      const result = await response.json();
      setEvents(result.data);
    } catch (err) {
      console.error('Error fetching events:', err);
    }
  };

  const handleDateRangeChange = (newRange) => {
    setDateRange(newRange);
  };

  const handleEventTypeChange = (eventType) => {
    setSelectedEventType(eventType);
  };

  const handleEventHighlight = (event) => {
    setHighlightedEvent(event);
  };

  if (error) {
    return (
      <div className="App error-container">
        <div className="error-message">
          <h2>⚠️ Error</h2>
          <p>{error}</p>
          <p className="error-hint">
            Make sure the Flask backend is running on port 5000
          </p>
          <button onClick={() => window.location.reload()}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="App">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <h1>📊 Brent Oil Price Analysis Dashboard</h1>
          <p>Change Point Detection & Event Impact Analysis</p>
        </div>
        <div className="header-subtitle">
           Data-Driven Insights for the Energy Sector
        </div>
      </header>

      {/* Main Content */}
      <div className="main-container">
        {/* Stats Cards */}
        {stats && <StatsCards stats={stats} />}

        {/* Filters */}
        <div className="filters-section">
          <DateRangeFilter 
            dateRange={dateRange}
            onDateRangeChange={handleDateRangeChange}
          />
          
          <div className="event-type-filter">
            <label>Event Type:</label>
            <select 
              value={selectedEventType} 
              onChange={(e) => handleEventTypeChange(e.target.value)}
            >
              <option value="all">All Events</option>
              <option value="OPEC Policy">OPEC Policy</option>
              <option value="Geopolitical Conflict">Geopolitical Conflict</option>
              <option value="Economic Crisis">Economic Crisis</option>
              <option value="Military Conflict">Military Conflict</option>
              <option value="Economic Sanctions">Economic Sanctions</option>
              <option value="Supply Disruption">Supply Disruption</option>
              <option value="Pandemic">Pandemic</option>
            </select>
          </div>
        </div>

        {/* Main Chart */}
        <div className="chart-section">
          <h2>Historical Price Analysis</h2>
          {loading ? (
            <div className="loading">Loading price data...</div>
          ) : (
            <PriceChart 
              prices={prices} 
              events={events}
              highlightedEvent={highlightedEvent}
            />
          )}
        </div>

        {/* Two Column Layout */}
        <div className="two-column-layout">
          {/* Events Panel */}
          <div className="panel">
            <h2>Major Events</h2>
            <EventsPanel 
              events={events}
              onEventHighlight={handleEventHighlight}
              highlightedEvent={highlightedEvent}
            />
          </div>

          {/* Volatility Analysis */}
          <div className="panel">
            <h2>Volatility Analysis</h2>
            <VolatilityChart prices={prices} />
          </div>
        </div>

        {/* Event Correlation */}
        <div className="chart-section">
          <h2>Event Impact Correlation</h2>
          <EventCorrelation events={events} />
        </div>

        {/* Footer */}
        <footer className="app-footer">
          <p>
            Data Source: Brent Oil Prices (1987-2022) | 
            Analysis: Bayesian Change Point Detection with PyMC | 
            © 2026 Birhan Energies
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;
