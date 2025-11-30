import React from "react";

export default function StatusBar({ error }) {
  return (
    <div className="status-bar">
      {error ? (
        <span className="status-error">Error: {error}</span>
      ) : (
        <span className="status-ok">Backend: OK (polling updates)</span>
      )}
    </div>
  );
}
