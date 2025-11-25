import React from "react";

export default function ControlPanel({ config, onChange, onSubmit }) {
  return (
    <div className="control-panel">
      <div className="field">
        <label>Количество курьеров</label>
        <input
          type="number"
          min="1"
          max="100"
          value={config.num_couriers}
          onChange={(e) => onChange("num_couriers", e.target.value)}
        />
      </div>
      <div className="field">
        <label>Скорость курьеров (м/с)</label>
        <input
          type="number"
          min="0.1"
          step="0.1"
          value={config.courier_speed}
          onChange={(e) => onChange("courier_speed", e.target.value)}
        />
      </div>
      <div className="field">
        <label>Заказов в минуту</label>
        <input
          type="number"
          min="0"
          step="0.1"
          value={config.orders_per_minute}
          onChange={(e) => onChange("orders_per_minute", e.target.value)}
        />
      </div>
      <button className="btn" onClick={onSubmit}>
        Применить
      </button>
    </div>
  );
}
