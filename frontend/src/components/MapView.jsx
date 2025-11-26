import React, { useState, useRef, useEffect } from "react";
import { getStaticImageUrl } from "../api";

const MAP_SIZE = 900;
const MIN_SCALE = 0.3;
const MAX_SCALE = 4.0;
const ZOOM_SENSITIVITY = 0.002; // Увеличена чувствительность для более плавного зума
const EXTRA_PAN_SPACE = 3000; // Дополнительное пространство для панорамирования вправо и вниз (на максимальном зуме)

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
  const hasDragged = useRef(false); // Флаг для отслеживания перетаскивания

  // Ограничение смещения, чтобы карта не уходила слишком далеко
  const constrainOffset = (newOffset, currentScale) => {
    const container = containerRef.current;
    if (!container) return newOffset;

    const containerRect = container.getBoundingClientRect();
    const scaledMapSize = MAP_SIZE * currentScale;

    // Дополнительное пространство масштабируется с уровнем зума
    // На максимальном зуме даем больше места для навигации по краям
    // Используем квадратичную зависимость для более плавного увеличения на больших масштабах
    const scaleFactor = Math.pow(currentScale / MAX_SCALE, 1.5);
    const extraSpace = EXTRA_PAN_SPACE * scaleFactor;

    // Если карта меньше контейнера, центрируем её
    if (scaledMapSize <= containerRect.width) {
      newOffset.x = (containerRect.width - scaledMapSize) / 2;
    } else {
      const maxOffsetX = (scaledMapSize - containerRect.width) / 2;
      // Расширяем границы для движения влево (отрицательное x) и вправо (положительное x)
      // Позволяем сдвинуть карту так, чтобы можно было пройтись по всем краям
      const minOffsetXExtended = -(scaledMapSize - containerRect.width) - extraSpace;
      const maxOffsetXExtended = scaledMapSize - containerRect.width + extraSpace;
      newOffset.x = Math.max(minOffsetXExtended, Math.min(maxOffsetXExtended, newOffset.x));
    }

    if (scaledMapSize <= containerRect.height) {
      newOffset.y = (containerRect.height - scaledMapSize) / 2;
    } else {
      const maxOffsetY = (scaledMapSize - containerRect.height) / 2;
      // Расширяем границы для движения вверх (отрицательное y) и вниз (положительное y)
      // Позволяем сдвинуть карту так, чтобы можно было пройтись по всем краям
      const minOffsetYExtended = -(scaledMapSize - containerRect.height) - extraSpace;
      const maxOffsetYExtended = scaledMapSize - containerRect.height + extraSpace;
      newOffset.y = Math.max(minOffsetYExtended, Math.min(maxOffsetYExtended, newOffset.y));
    }

    return newOffset;
  };

  // Зум к точке курсора и панорамирование тачпадом
  const handleWheel = (e) => {
    e.preventDefault();
    const container = containerRef.current;
    if (!container) return;

    // Определяем, это зум или панорамирование
    // Зум: Ctrl/Cmd зажат (пинч на Mac), есть deltaZ, или только вертикальная прокрутка (колесико мыши)
    // Панорамирование: есть значительная горизонтальная прокрутка без Ctrl/Cmd (двухпальцевая прокрутка на тачпаде)
    const isZoomGesture = e.ctrlKey || e.metaKey || Math.abs(e.deltaZ) > 0;
    const hasSignificantHorizontalScroll = Math.abs(e.deltaX) > 10 && Math.abs(e.deltaX) > Math.abs(e.deltaY) * 0.7;
    const isPan = hasSignificantHorizontalScroll && !isZoomGesture;

    if (isPan) {
      // Двухпальцевое панорамирование на тачпаде (горизонтальная/вертикальная прокрутка)
      setOffset((prev) => constrainOffset(
        {
          x: prev.x - e.deltaX,
          y: prev.y - e.deltaY,
        },
        scale
      ));
    } else {
      // Зум колесиком мыши или пинч-жестом на тачпаде
      const rect = container.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      // Вычисляем точку на карте до зума
      const pointBeforeZoom = {
        x: (mouseX - offset.x) / scale,
        y: (mouseY - offset.y) / scale,
      };

      // Вычисляем новый масштаб
      // Для приближения: deltaY < 0 (прокрутка вверх или разведение пальцев на тачпаде)
      // Для отдаления: deltaY > 0 (прокрутка вниз или сведение пальцев на тачпаде)
      // На тачпаде с Ctrl/Cmd: отрицательный deltaY = разведение = приближение
      const zoomDelta = e.deltaY * ZOOM_SENSITIVITY;
      const zoomFactor = 1 - zoomDelta;
      let newScale = scale * zoomFactor;

      // Ограничиваем масштаб
      if (newScale < MIN_SCALE) newScale = MIN_SCALE;
      if (newScale > MAX_SCALE) newScale = MAX_SCALE;

      // Вычисляем новое смещение, чтобы точка под курсором осталась на месте
      const newOffset = {
        x: mouseX - pointBeforeZoom.x * newScale,
        y: mouseY - pointBeforeZoom.y * newScale,
      };

      setScale(newScale);
      setOffset((prev) => constrainOffset(newOffset, newScale));
    }
  };

  const handleMouseDown = (e) => {
    // Разрешаем перетаскивание везде, кроме кликов на интерактивные элементы
    // Интерактивные элементы: курьеры (circle с классом courier-dot или courier-selected)
    const target = e.target;
    const isCourier =
      (target.tagName === 'circle' &&
        (target.classList.contains('courier-dot') ||
          target.classList.contains('courier-selected'))) ||
      (target.tagName === 'text' && target.closest('g')?.querySelector('circle.courier-dot, circle.courier-selected'));

    // Если клик на курьере, не начинаем перетаскивание (разрешаем выбор курьера)
    if (isCourier) {
      return;
    }

    // Предотвращаем выделение текста при перетаскивании
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

    // Если перемещение больше 3 пикселей, считаем это перетаскиванием
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
      // Если было перетаскивание, предотвращаем клик
      if (hasDragged.current) {
        e.preventDefault();
        e.stopPropagation();
      }
      hasDragged.current = false;
    }
  };

  // Поддержка жестов тачпада (двумя пальцами)
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
      // Одиночное касание для панорамирования
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

        // Вычисляем масштаб на основе изменения расстояния между пальцами
        const scaleChange = distance / lastTouchDistance.current;
        let newScale = touchStartRef.current.scale * scaleChange;
        newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, newScale));

        // Используем текущие значения offset и scale для более точных вычислений
        const currentOffset = offset;
        const currentScale = scale;

        // Вычисляем точку на карте до зума (относительно начального состояния жеста)
        const pointBeforeZoom = {
          x: (centerX - touchStartRef.current.offset.x) / touchStartRef.current.scale,
          y: (centerY - touchStartRef.current.offset.y) / touchStartRef.current.scale,
        };

        // Вычисляем новое смещение, чтобы точка между пальцами осталась на месте
        const newOffset = {
          x: centerX - pointBeforeZoom.x * newScale,
          y: centerY - pointBeforeZoom.y * newScale,
        };

        setScale(newScale);
        setOffset((prev) => {
          const constrained = constrainOffset(newOffset, newScale);
          // Обновляем touchStartRef для следующего движения
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
    // Игнорируем клик, если было перетаскивание
    if (hasDragged.current) {
      e.preventDefault();
      e.stopPropagation();
      return;
    }
    if (onSelectCourier) {
      onSelectCourier(null);
    }
  };

  // Центрируем карту при первой загрузке
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
  }, []); // Только при монтировании компонента


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

            // NEW: аккуратный tooltip с учётом загрузки и веса заказа
            const currentLoad = (c.current_load || 0).toFixed(1);
            const maxCapacity = (c.max_capacity || 0).toFixed(1);
            const tooltip = order
              ? `Курьер #${c.id}
Состояние: ${c.state}
Загрузка: ${currentLoad} / ${maxCapacity} кг
Текущий заказ #${order.id}, статус: ${order.status}, вес: ${(order.weight || 0).toFixed(1)} кг`
              : `Курьер #${c.id}
Состояние: ${c.state}
Загрузка: ${currentLoad} / ${maxCapacity} кг
Текущий заказ: нет`;

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
