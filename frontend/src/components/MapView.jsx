import React, { useState, useRef, useEffect } from "react";
import { getStaticImageUrl } from "../api";

const MAP_SIZE = 900;
const MIN_SCALE = 0.3;
const MAX_SCALE = 4.0;
const ZOOM_SENSITIVITY = 0.002; // Increased sensitivity for smoother zoom
const EXTRA_PAN_SPACE = 3000; // Extra space for panning right and down (at maximum zoom)

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
  const containerRef = useRef(null);
  const touchStartRef = useRef(null);
  const lastTouchDistance = useRef(0);
  const isPinching = useRef(false);
  const hasDragged = useRef(false); // Flag for tracking dragging

  // Constrain offset so map doesn't go too far
  const constrainOffset = (newOffset, currentScale) => {
    const container = containerRef.current;
    if (!container) return newOffset;

    const containerRect = container.getBoundingClientRect();
    const scaledMapSize = MAP_SIZE * currentScale;

    // Extra space scales with zoom level
    // At maximum zoom, give more space for edge navigation
    // Use quadratic dependency for smoother increase at large scales
    const scaleFactor = Math.pow(currentScale / MAX_SCALE, 1.5);
    const extraSpace = EXTRA_PAN_SPACE * scaleFactor;

    // If map is smaller than container, center it
    if (scaledMapSize <= containerRect.width) {
      newOffset.x = (containerRect.width - scaledMapSize) / 2;
    } else {
      const maxOffsetX = (scaledMapSize - containerRect.width) / 2;
      // Extend boundaries for movement left (negative x) and right (positive x)
      // Allow map shift so we can navigate all edges
      const minOffsetXExtended = -(scaledMapSize - containerRect.width) - extraSpace;
      const maxOffsetXExtended = scaledMapSize - containerRect.width + extraSpace;
      newOffset.x = Math.max(minOffsetXExtended, Math.min(maxOffsetXExtended, newOffset.x));
    }

    if (scaledMapSize <= containerRect.height) {
      newOffset.y = (containerRect.height - scaledMapSize) / 2;
    } else {
      const maxOffsetY = (scaledMapSize - containerRect.height) / 2;
      // Extend boundaries for movement up (negative y) and down (positive y)
      // Allow map shift so we can navigate all edges
      const minOffsetYExtended = -(scaledMapSize - containerRect.height) - extraSpace;
      const maxOffsetYExtended = scaledMapSize - containerRect.height + extraSpace;
      newOffset.y = Math.max(minOffsetYExtended, Math.min(maxOffsetYExtended, newOffset.y));
    }

    return newOffset;
  };

  // Zoom to cursor point and trackpad panning
  const handleWheel = (e) => {
    e.preventDefault();
    const container = containerRef.current;
    if (!container) return;

    // Determine if this is zoom or panning
    // Zoom: Ctrl/Cmd held (pinch on Mac), has deltaZ, or only vertical scroll (mouse wheel)
    // Panning: has significant horizontal scroll without Ctrl/Cmd (two-finger scroll on trackpad)
    const isZoomGesture = e.ctrlKey || e.metaKey || Math.abs(e.deltaZ) > 0;
    const hasSignificantHorizontalScroll = Math.abs(e.deltaX) > 10 && Math.abs(e.deltaX) > Math.abs(e.deltaY) * 0.7;
    const isPan = hasSignificantHorizontalScroll && !isZoomGesture;

    if (isPan) {
      // Two-finger panning on trackpad (horizontal/vertical scroll)
      setOffset((prev) => constrainOffset(
        {
          x: prev.x - e.deltaX,
          y: prev.y - e.deltaY,
        },
        scale
      ));
    } else {
      // Zoom with mouse wheel or pinch gesture on trackpad
      const rect = container.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      // Calculate point on map before zoom
      const pointBeforeZoom = {
        x: (mouseX - offset.x) / scale,
        y: (mouseY - offset.y) / scale,
      };

      // Calculate new scale
      // For zoom in: deltaY < 0 (scroll up or pinch out on trackpad)
      // For zoom out: deltaY > 0 (scroll down or pinch in on trackpad)
      // On trackpad with Ctrl/Cmd: negative deltaY = pinch out = zoom in
      const zoomDelta = e.deltaY * ZOOM_SENSITIVITY;
      const zoomFactor = 1 - zoomDelta;
      let newScale = scale * zoomFactor;

      // Constrain scale
      if (newScale < MIN_SCALE) newScale = MIN_SCALE;
      if (newScale > MAX_SCALE) newScale = MAX_SCALE;

      // Calculate new offset so point under cursor stays in place
      const newOffset = {
        x: mouseX - pointBeforeZoom.x * newScale,
        y: mouseY - pointBeforeZoom.y * newScale,
      };

      setScale(newScale);
      setOffset((prev) => constrainOffset(newOffset, newScale));
    }
  };

  const handleMouseDown = (e) => {
    // Allow dragging everywhere except clicks on interactive elements
    // Interactive elements: couriers (circle with class courier-dot or courier-selected)
    const target = e.target;
    const isCourier =
      (target.tagName === 'circle' &&
        (target.classList.contains('courier-dot') ||
          target.classList.contains('courier-selected'))) ||
      (target.tagName === 'text' && target.closest('g')?.querySelector('circle.courier-dot, circle.courier-selected'));

    // If click on courier, don't start dragging (allow courier selection)
    if (isCourier) {
      return;
    }

    // Prevent text selection during dragging
    e.preventDefault();
    dragging.current = true;
    hasDragged.current = false;
    lastPos.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e) => {
    if (!dragging.current) return;
    e.preventDefault();
    const dx = e.clientX - lastPos.current.x;
    const dy = e.clientY - lastPos.current.y;

    // If movement is more than 3 pixels, consider it dragging
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
      hasDragged.current = true;
    }

    lastPos.current = { x: e.clientX, y: e.clientY };
    setOffset((prev) => constrainOffset(
      { x: prev.x + dx, y: prev.y + dy },
      scale
    ));
  };

  const handleMouseUp = (e) => {
    if (dragging.current) {
      dragging.current = false;
      // If there was dragging, prevent click
      if (hasDragged.current) {
        e.preventDefault();
        e.stopPropagation();
      }
      hasDragged.current = false;
    }
  };

  // Trackpad gesture support (two fingers)
  const getTouchDistance = (touch1, touch2) => {
    const dx = touch1.clientX - touch2.clientX;
    const dy = touch1.clientY - touch2.clientY;
    return Math.sqrt(dx * dx + dy * dy);
  };

  const getTouchCenter = (touch1, touch2) => {
    return {
      x: (touch1.clientX + touch2.clientX) / 2,
      y: (touch1.clientY + touch2.clientY) / 2,
    };
  };

  const handleTouchStart = (e) => {
    if (e.touches.length === 2) {
      e.preventDefault();
      isPinching.current = true;
      const touch1 = e.touches[0];
      const touch2 = e.touches[1];
      lastTouchDistance.current = getTouchDistance(touch1, touch2);
      const center = getTouchCenter(touch1, touch2);
      touchStartRef.current = {
        center,
        offset: { ...offset },
        scale: scale,
      };
    } else if (e.touches.length === 1) {
      // Single touch for panning
      dragging.current = true;
      lastPos.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
    }
  };

  const handleTouchMove = (e) => {
    if (e.touches.length === 2 && isPinching.current) {
      e.preventDefault();
      const touch1 = e.touches[0];
      const touch2 = e.touches[1];
      const distance = getTouchDistance(touch1, touch2);
      const center = getTouchCenter(touch1, touch2);

      if (touchStartRef.current) {
        const container = containerRef.current;
        if (!container) return;

        const rect = container.getBoundingClientRect();
        const centerX = center.x - rect.left;
        const centerY = center.y - rect.top;

        // Calculate scale based on change in distance between fingers
        const scaleChange = distance / lastTouchDistance.current;
        let newScale = touchStartRef.current.scale * scaleChange;
        newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, newScale));

        // Use current offset and scale values for more accurate calculations
        const currentOffset = offset;
        const currentScale = scale;

        // Calculate point on map before zoom (relative to initial gesture state)
        const pointBeforeZoom = {
          x: (centerX - touchStartRef.current.offset.x) / touchStartRef.current.scale,
          y: (centerY - touchStartRef.current.offset.y) / touchStartRef.current.scale,
        };

        // Calculate new offset so point between fingers stays in place
        const newOffset = {
          x: centerX - pointBeforeZoom.x * newScale,
          y: centerY - pointBeforeZoom.y * newScale,
        };

        setScale(newScale);
        setOffset((prev) => {
          const constrained = constrainOffset(newOffset, newScale);
          // Update touchStartRef for next movement
          if (touchStartRef.current) {
            touchStartRef.current.offset = constrained;
            touchStartRef.current.scale = newScale;
          }
          return constrained;
        });
        lastTouchDistance.current = distance;
      }
    } else if (e.touches.length === 1 && dragging.current) {
      e.preventDefault();
      const dx = e.touches[0].clientX - lastPos.current.x;
      const dy = e.touches[0].clientY - lastPos.current.y;
      lastPos.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      setOffset((prev) => constrainOffset(
        { x: prev.x + dx, y: prev.y + dy },
        scale
      ));
    }
  };

  const handleTouchEnd = (e) => {
    if (e.touches.length < 2) {
      isPinching.current = false;
      touchStartRef.current = null;
    }
    if (e.touches.length === 0) {
      dragging.current = false;
    }
  };

  const handleBackgroundClick = (e) => {
    // Ignore click if there was dragging
    if (hasDragged.current) {
      e.preventDefault();
      e.stopPropagation();
      return;
    }
    if (onSelectCourier) {
      onSelectCourier(null);
    }
  };

  // Center map on first load
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const containerRect = container.getBoundingClientRect();
    const scaledMapSize = MAP_SIZE * scale;

    if (scaledMapSize <= containerRect.width) {
      setOffset(prev => ({
        ...prev,
        x: (containerRect.width - scaledMapSize) / 2
      }));
    }

    if (scaledMapSize <= containerRect.height) {
      setOffset(prev => ({
        ...prev,
        y: (containerRect.height - scaledMapSize) / 2
      }));
    }
  }, []); // Only on component mount


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
      ref={containerRef}
      className={`map-view ${dragging.current ? 'dragging' : ''}`}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseUp}
      onMouseUp={handleMouseUp}
      onClick={handleBackgroundClick}
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
      style={{ touchAction: 'none', userSelect: 'none' }}
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

          {/* highlight selected courier route */}
          {selectedPathPolyline}

          {/* highlight shop and house */}
          {highlightShop}
          {highlightHouse}

          {/* houses */}
          {houses.map((h) => (
            <circle
              key={`house-${h.id}`}
              cx={h.x}
              cy={h.y}
              r={3}
              className="house-dot"
            >
              <title>House #{h.id}</title>
            </circle>
          ))}

          {/* shops */}
          {shops.map((s) => (
            <rect
              key={`shop-${s.id}`}
              x={s.x - 4}
              y={s.y - 4}
              width={8}
              height={8}
              className="shop-dot"
            >
              <title>Shop #{s.id}</title>
            </rect>
          ))}

          {/* couriers */}
          {couriers.map((c) => {
            const selected = c.id === selectedCourierId;
            const order = c.current_order_id
              ? orderById.get(c.current_order_id)
              : null;

            // NEW: careful tooltip with load and order weight
            const currentLoad = (c.current_load || 0).toFixed(1);
            const maxCapacity = (c.max_capacity || 0).toFixed(1);
            const tooltip = order
              ? `Courier #${c.id}
State: ${c.state}
Load: ${currentLoad} / ${maxCapacity} kg
Current order #${order.id}, status: ${order.status}, weight: ${(order.weight || 0).toFixed(1)} kg`
              : `Courier #${c.id}
State: ${c.state}
Load: ${currentLoad} / ${maxCapacity} kg
Current order: none`;

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
                <title>{tooltip}</title>
              </g>
            );
          })}
        </g>
      </svg>
    </div>
  );
}
