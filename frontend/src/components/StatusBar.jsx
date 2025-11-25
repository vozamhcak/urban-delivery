import React from "react";

export default function StatusBar({ error }) {
  return (
    <div className="status-bar">
      {error ? (
        <span className="status-error">Ошибка: {error}</span>
      ) : (
        <span className="status-ok">Backend: OK (обновление по опросу)</span>
      )}
    </div>
  );
}
