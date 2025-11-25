import React, { useEffect, useState } from "react";
import { fetchState, fetchConfig, updateConfig } from "./api";
import MapView from "./components/MapView";
import ControlPanel from "./components/ControlPanel";
import LogPanel from "./components/LogPanel";
import StatusBar from "./components/StatusBar";

const POLL_INTERVAL_MS = 500;

export default function App() {
  const [state, setState] = useState(null);
  const [error, setError] = useState(null);

  const [configDraft, setConfigDraft] = useState({
    num_couriers: 5,
    courier_speed: 1.5,
    orders_per_minute: 2.0,
  });

  const [selectedCourierId, setSelectedCourierId] = useState(null);

  // поллинг состояния симуляции
  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const data = await fetchState();
        if (!cancelled) {
          setState(data);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e.message || "Ошибка сервера");
        }
      }
      if (!cancelled) {
        setTimeout(poll, POLL_INTERVAL_MS);
      }
    }

    poll();
    return () => {
      cancelled = true;
    };
  }, []);

  // отдельная загрузка конфига (не зависит от поллинга)
  useEffect(() => {
    let cancelled = false;

    async function loadConfig() {
      try {
        const cfg = await fetchConfig();
        if (!cancelled) {
          setConfigDraft({
            num_couriers: cfg.num_couriers,
            courier_speed: cfg.courier_speed,
            orders_per_minute: cfg.orders_per_minute,
          });
        }
      } catch (e) {
        if (!cancelled) {
          setError(e.message || "Не удалось загрузить конфигурацию");
        }
      }
    }

    loadConfig();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleConfigChange = (field, value) => {
    setConfigDraft((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleConfigSubmit = async () => {
    try {
      const cfg = {
        num_couriers: Number(configDraft.num_couriers),
        courier_speed: Number(configDraft.courier_speed),
        orders_per_minute: Number(configDraft.orders_per_minute),
      };
      const updated = await updateConfig(cfg);
      // после обновления синхронизируем черновик с подтверждённым конфигом
      setConfigDraft({
        num_couriers: updated.num_couriers,
        courier_speed: updated.courier_speed,
        orders_per_minute: updated.orders_per_minute,
      });
    } catch (e) {
      setError(e.message || "Не удалось обновить конфигурацию");
    }
  };

  return (
    <div className="app-root">
      <div className="top-layout">
        <div className="map-container">
          {state ? (
            <MapView
              houses={state.houses}
              shops={state.shops}
              couriers={state.couriers}
              orders={state.orders}
              roadNodes={state.road_nodes}
              selectedCourierId={selectedCourierId}
              onSelectCourier={setSelectedCourierId}
            />
          ) : (
            <div className="centered-message">
              Загрузка состояния симуляции...
            </div>
          )}
        </div>
        <div className="sidebar">
          <h2>Настройки симуляции</h2>
          <ControlPanel
            config={configDraft}
            onChange={handleConfigChange}
            onSubmit={handleConfigSubmit}
          />
        </div>
      </div>
      <div className="log-area">
        <LogPanel logs={state ? state.logs : []} />
      </div>
      <StatusBar error={error} />
    </div>
  );
}
