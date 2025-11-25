const API_URL = "http://localhost:8000";

export async function fetchState() {
  const res = await fetch(`${API_URL}/api/state`);
  if (!res.ok) {
    throw new Error("Failed to fetch state");
  }
  return res.json();
}

export async function fetchConfig() {
  const res = await fetch(`${API_URL}/api/config`);
  if (!res.ok) {
    throw new Error("Failed to fetch config");
  }
  return res.json();
}

export async function updateConfig(config) {
  const res = await fetch(`${API_URL}/api/config`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!res.ok) {
    throw new Error("Failed to update config");
  }
  return res.json();
}

export function getStaticImageUrl() {
  return `${API_URL}/static/ref.png`;
}
