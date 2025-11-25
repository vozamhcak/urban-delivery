import React, { useState, useRef } from "react";
import { getStaticImageUrl } from "../api";

const MAP_SIZE = 900; 

export default function MapView({
  houses,
  shops,
  couriers,
  orders,
  roadNodes,
  selectedCourierId,
  onSelectCourier,
}) {
  const [scale, setScale] = useState(1.0);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const dragging = useRef(false);
  const lastPos = useRef({ x: 0, y: 0 });

  const handleWheel = (e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    let newScale = scale + delta;
    if (newScale < 0.3) newScale = 0.3;
    if (newScale > 4.0) newScale = 4.0;
    setScale(newScale);
  };

  const handleMouseDown = (e) => {
    dragging.current = true;
    lastPos.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e) => {
    if (!dragging.current) return;
    const dx = e.clientX - lastPos.current.x;
    const dy = e.clientY - lastPos.current.y;
    lastPos.current = { x: e.clientX, y: e.clientY };
    setOffset((prev) => ({ x: prev.x + dx, y: prev.y + dy }));
  };

  const handleMouseUp = () => {
    dragging.current = false;
  };

  const handleBackgroundClick = () => {
    if (onSelectCourier) {
      onSelectCourier(null);
    }
  };

  
  const activeOrders = orders.filter((o) => o.status !== "done");
  const orderById = new Map(activeOrders.map((o) => [o.id, o]));

  const selectedCourier = couriers.find(
    (c) => c.id === selectedCourierId
  );

  const nodesById = {};
  (roadNodes || []).forEach((n) => {
    nodesById[n.id] = n;
  });

  
  let selectedPathPolyline = null;
  let highlightShop = null;
  let highlightHouse = null;

  if (selectedCourier) {
    
    const pathNodes = (selectedCourier.path || [])
      .map((id) => nodesById[id])
      .filter(Boolean);

    if (pathNodes.length > 1) {
      const points = pathNodes.map((n) => `${n.x},${n.y}`).join(" ");
      selectedPathPolyline = (
        <polyline
          points={points}
          className="courier-path-line"
        />
      );
    }

    const order = selectedCourier.current_order_id
      ? orders.find((o) => o.id === selectedCourier.current_order_id)
      : null;

    if (order) {
      const shop = shops.find((s) => s.id === order.shop_id);
      const house = houses.find((h) => h.id === order.house_id);

      if (shop) {
        highlightShop = (
          <circle
            cx={shop.x}
            cy={shop.y}
            r={10}
            className="highlight-shop"
          />
        );
      }

      if (house) {
        highlightHouse = (
          <circle
            cx={house.x}
            cy={house.y}
            r={10}
            className="highlight-house"
          />
        );
      }
    }
  }

  return (
    <div
      className="map-view"
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseUp}
      onMouseUp={handleMouseUp}
      onClick={handleBackgroundClick}
    >
      <svg
        width="100%"
        height="100%"
        viewBox={`0 0 ${MAP_SIZE} ${MAP_SIZE}`}
        preserveAspectRatio="xMidYMid meet"
      >
        <g transform={`translate(${offset.x},${offset.y}) scale(${scale})`}>
          <image
            href={getStaticImageUrl()}
            x="0"
            y="0"
            width={MAP_SIZE}
            height={MAP_SIZE}
          />

          {/* подсветка маршрута выбранного курьера */}
          {selectedPathPolyline}

          {/* подсветка магазина и дома */}
          {highlightShop}
          {highlightHouse}

          {/* дома */}
          {houses.map((h) => (
            <circle
              key={`house-${h.id}`}
              cx={h.x}
              cy={h.y}
              r={3}
              className="house-dot"
            >
              <title>Дом #{h.id}</title>
            </circle>
          ))}

          {/* магазины */}
          {shops.map((s) => (
            <rect
              key={`shop-${s.id}`}
              x={s.x - 4}
              y={s.y - 4}
              width={8}
              height={8}
              className="shop-dot"
            >
              <title>Магазин #{s.id}</title>
            </rect>
          ))}

          {/* курьеры */}
          {couriers.map((c) => {
            const selected = c.id === selectedCourierId;
            const order = c.current_order_id
              ? orderById.get(c.current_order_id)
              : null;

            return (
              <g key={`courier-${c.id}`}>
                <circle
                  cx={c.x}
                  cy={c.y}
                  r={selected ? 7 : 5}
                  className={selected ? "courier-selected" : "courier-dot"}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (onSelectCourier) {
                      onSelectCourier(c.id);
                    }
                  }}
                />
                <text
                  x={c.x + 6}
                  y={c.y - 6}
                  className="courier-label"
                >
                  {c.id}
                </text>
                {order && (
                  <title>
                    Курьер #{c.id}, заказ #{order.id}, статус {order.status}
                  </title>
                )}
              </g>
            );
          })}
        </g>
      </svg>
    </div>
  );
}
