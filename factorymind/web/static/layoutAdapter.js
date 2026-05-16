export function buildApplyLayoutAction(parsed, options = {}) {
  return {
    type: "apply_layout",
    layout: parsed,
    worker_count: clampInt(options.workerCount, 1, 24, 3),
    order_volume: clampInt(options.orderVolume, 1, 20, 6),
  };
}

export function exportLayoutJson(parsed) {
  return JSON.stringify(parsed, null, 2);
}

export function downloadJson(filename, jsonText) {
  const blob = new Blob([jsonText], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function readJsonFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        resolve(JSON.parse(String(reader.result || "{}")));
      } catch (error) {
        reject(error);
      }
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
}

function clampInt(value, min, max, fallback) {
  const parsed = Number.parseInt(value, 10);
  if (Number.isNaN(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
}
