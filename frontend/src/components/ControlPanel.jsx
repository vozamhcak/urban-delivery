import React from "react";

export default function ControlPanel({ config, onChange, onSubmit }) {
  return (
    <div className="control-panel">
      <div className="field">
        <label>Number of couriers</label>
        <input
          type="number"
          min="1"
          max="100"
          value={config.num_couriers}
          onChange={(e) => onChange("num_couriers", e.target.value)}
        />
      </div>
      <div className="field">
        <label>Courier speed (m/s)</label>
        <input
          type="number"
          min="0.1"
          step="0.1"
          value={config.courier_speed}
          onChange={(e) => onChange("courier_speed", e.target.value)}
        />
      </div>
      <div className="field">
        <label>Orders per minute</label>
        <input
          type="number"
          min="0"
          step="0.1"
          value={config.orders_per_minute}
          onChange={(e) => onChange("orders_per_minute", e.target.value)}
        />
      </div>
      <button className="btn" onClick={onSubmit}>
        Apply
      </button>
    </div>
  );
}
