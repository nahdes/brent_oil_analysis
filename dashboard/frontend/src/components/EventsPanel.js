import React, { useState } from 'react';
import './EventsPanel.css';

const EventsPanel = ({ events, onEventHighlight, highlightedEvent }) => {
  const [expandedEvent, setExpandedEvent] = useState(null);

  if (!events || events.length === 0) {
    return <div className="no-events">No events in selected date range</div>;
  }

  const getEventTypeColor = (eventType) => {
    const colors = {
      'OPEC Policy': '#3b82f6',
      'Geopolitical Conflict': '#ef4444',
      'Economic Crisis': '#f59e0b',
      'Military Conflict': '#dc2626',
      'Economic Sanctions': '#8b5cf6',
      'Supply Disruption': '#ec4899',
      'Pandemic': '#14b8a6'
    };
    return colors[eventType] || '#6b7280';
  };

  const handleEventClick = (event) => {
    if (expandedEvent === event.Date) {
      setExpandedEvent(null);
      onEventHighlight(null);
    } else {
      setExpandedEvent(event.Date);
      onEventHighlight(event);
    }
  };

  return (
    <div className="events-panel">
      <div className="events-list">
        {events.map((event, idx) => (
          <div
            key={idx}
            className={`event-item ${
              highlightedEvent && highlightedEvent.Date === event.Date ? 'highlighted' : ''
            } ${expandedEvent === event.Date ? 'expanded' : ''}`}
            onClick={() => handleEventClick(event)}
          >
            <div className="event-header">
              <div
                className="event-type-badge"
                style={{ backgroundColor: getEventTypeColor(event.Event_Type) }}
              >
                {event.Event_Type}
              </div>
              <div className="event-date">{event.Date}</div>
            </div>
            
            <div className="event-description">
              {event.Description}
            </div>

            {expandedEvent === event.Date && (
              <div className="event-details">
                {event.Event_Category && (
                  <p><strong>Category:</strong> {event.Event_Category}</p>
                )}
                {event.Region && (
                  <p><strong>Region:</strong> {event.Region}</p>
                )}
                {event.Severity && (
                  <p><strong>Severity:</strong> {event.Severity}</p>
                )}
                {event.Impact_Direction && (
                  <p>
                    <strong>Expected Impact:</strong>{' '}
                    <span className={`impact-${event.Impact_Direction.toLowerCase()}`}>
                      {event.Impact_Direction}
                    </span>
                  </p>
                )}
                {event.Notes && (
                  <p className="event-notes">{event.Notes}</p>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="events-summary">
        <p>Showing {events.length} event{events.length !== 1 ? 's' : ''}</p>
        <p className="help-text">Click on an event to highlight it on the chart</p>
      </div>
    </div>
  );
};

export default EventsPanel;
