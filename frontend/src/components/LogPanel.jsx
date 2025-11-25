import React from "react";

export default function LogPanel({ logs }) {
  return (
    <div className="log-panel">
      <h3>Логи</h3>
      <div className="log-list">
        {logs && logs.length > 0 ? (
          logs
            .slice()
            .sort((a, b) => a.ts - b.ts)
            .map((log, idx) => (
              <div key={idx} className="log-entry">
                <span className="log-time">
                  {new Date(log.ts * 1000).toLocaleTimeString()}
                </span>
                <span className="log-message">{log.message}</span>
              </div>
            ))
        ) : (
          <div className="log-empty">Пока нет событий</div>
        )}
      </div>
    </div>
  );
}
