import React from 'react';
import './DateRangeFilter.css';

const DateRangeFilter = ({ dateRange, onDateRangeChange }) => {
  const handleStartDateChange = (e) => {
    onDateRangeChange({
      ...dateRange,
      start: e.target.value
    });
  };

  const handleEndDateChange = (e) => {
    onDateRangeChange({
      ...dateRange,
      end: e.target.value
    });
  };

  const setPresetRange = (range) => {
    const end = new Date().toISOString().split('T')[0];
    let start;

    switch (range) {
      case '1y':
        start = new Date(new Date().setFullYear(new Date().getFullYear() - 1))
          .toISOString().split('T')[0];
        break;
      case '5y':
        start = new Date(new Date().setFullYear(new Date().getFullYear() - 5))
          .toISOString().split('T')[0];
        break;
      case '10y':
        start = new Date(new Date().setFullYear(new Date().getFullYear() - 10))
          .toISOString().split('T')[0];
        break;
      case 'all':
        start = '1987-05-20';
        break;
      default:
        return;
    }

    onDateRangeChange({ start, end });
  };

  return (
    <div className="date-range-filter">
      <div className="date-inputs">
        <div className="date-input-group">
          <label htmlFor="start-date">From:</label>
          <input
            id="start-date"
            type="date"
            value={dateRange.start}
            onChange={handleStartDateChange}
            max={dateRange.end}
          />
        </div>

        <span className="date-separator">→</span>

        <div className="date-input-group">
          <label htmlFor="end-date">To:</label>
          <input
            id="end-date"
            type="date"
            value={dateRange.end}
            onChange={handleEndDateChange}
            min={dateRange.start}
            max={new Date().toISOString().split('T')[0]}
          />
        </div>
      </div>

      <div className="preset-buttons">
        <button onClick={() => setPresetRange('1y')} className="preset-btn">
          1 Year
        </button>
        <button onClick={() => setPresetRange('5y')} className="preset-btn">
          5 Years
        </button>
        <button onClick={() => setPresetRange('10y')} className="preset-btn">
          10 Years
        </button>
        <button onClick={() => setPresetRange('all')} className="preset-btn">
          All Time
        </button>
      </div>
    </div>
  );
};

export default DateRangeFilter;
