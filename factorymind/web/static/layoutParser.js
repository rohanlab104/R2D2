export const TILE_COLORS = {
  floor: { hex: "#FFFFFF", rgb: [255, 255, 255] },
  wall: { hex: "#000000", rgb: [0, 0, 0] },
  shelf: { hex: "#2563EB", rgb: [37, 99, 235] },
  conveyor: { hex: "#16A34A", rgb: [22, 163, 74] },
  bin: { hex: "#F97316", rgb: [249, 115, 22] },
  charger: { hex: "#9333EA", rgb: [147, 51, 234] },
  restricted: { hex: "#DC2626", rgb: [220, 38, 38] },
  worker_spawn: { hex: "#EAB308", rgb: [234, 179, 8] },
  conveyor_red: { hex: "#7F1D1D", rgb: [127, 29, 29], packageColor: "Red" },
  conveyor_blue: { hex: "#1E3A8A", rgb: [30, 58, 138], packageColor: "Blue" },
  conveyor_green: { hex: "#14532D", rgb: [20, 83, 45], packageColor: "Green" },
  conveyor_yellow: { hex: "#713F12", rgb: [113, 63, 18], packageColor: "Yellow" },
  conveyor_magenta: { hex: "#581C87", rgb: [88, 28, 135], packageColor: "Magenta" },
  conveyor_cyan: { hex: "#164E63", rgb: [22, 78, 99], packageColor: "Cyan" },
  bin_red: { hex: "#FCA5A5", rgb: [252, 165, 165], packageColor: "Red" },
  bin_blue: { hex: "#93C5FD", rgb: [147, 197, 253], packageColor: "Blue" },
  bin_green: { hex: "#86EFAC", rgb: [134, 239, 172], packageColor: "Green" },
  bin_yellow: { hex: "#FEF08A", rgb: [254, 240, 138], packageColor: "Yellow" },
  bin_magenta: { hex: "#F0ABFC", rgb: [240, 171, 252], packageColor: "Magenta" },
  bin_cyan: { hex: "#A5F3FC", rgb: [165, 243, 252], packageColor: "Cyan" },
};

const OBJECT_TILES = new Set(Object.keys(TILE_COLORS).filter((tile) => tile !== "floor"));

function luminance(r, g, b) {
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

function sortedLuminancesFromImageData(data) {
  const arr = [];
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] < 32) continue;
    arr.push(luminance(data[i], data[i + 1], data[i + 2]));
  }
  arr.sort((a, b) => a - b);
  return arr;
}

function percentile(sorted, p) {
  if (!sorted.length) return 128;
  const idx = Math.min(
    sorted.length - 1,
    Math.max(0, Math.floor((sorted.length - 1) * p)),
  );
  return sorted[idx];
}

/** Skip color-matching on large flat blueprint paper / grid tint regions. */
function isLikelyBlueprintBackground(r, g, b, a) {
  if (a < 32) return true;
  const L = luminance(r, g, b);
  if (L >= 198) return true;
  const maxc = Math.max(r, g, b);
  const minc = Math.min(r, g, b);
  if (maxc - minc < 28 && L > 100) return true;
  // Cyan / blue sheet (dark or light)
  if (b >= r - 6 && b >= g - 6 && L > 88 && L < 218) return true;
  return false;
}

/**
 * Blueprint / line-art maps: infer wall vs walkable from ink vs paper, while
 * still picking up crisp colored regions (legend colors) as equipment tiles.
 */
function classifyBlueprintCell(r, g, b, a, lumas, tolerance, lineFold) {
  if (a < 32) return "floor";

  const maxc = Math.max(r, g, b);
  const minc = Math.min(r, g, b);
  const chroma = maxc - minc;
  const boostedTol = Number(tolerance) + 32;
  const useColor = chroma >= 44 && !isLikelyBlueprintBackground(r, g, b, a);

  if (useColor) {
    const tile = nearestTile([r, g, b], boostedTol);
    if (tile !== "floor") return tile;
  }

  const L = luminance(r, g, b);
  const p10 = percentile(lumas, 0.1);
  const p50 = percentile(lumas, 0.5);
  const p90 = percentile(lumas, 0.9);
  const sens = Math.min(0.62, Math.max(0.28, lineFold));
  // ink darker than paper
  if (p50 > 118) {
    const cut = p10 + (p50 - p10) * sens;
    return L <= cut ? "wall" : "floor";
  }
  // light strokes on dark / blue stock
  const cut = p90 - (p90 - p50) * sens;
  return L >= cut ? "wall" : "floor";
}

/** Map 0–100 UI “wall strength” to internal line fold (0.28–0.62). */
function optionsBlueprintLineFold(options) {
  const v = Number(options?.blueprintLineSensitivity ?? 45);
  if (!Number.isFinite(v)) return 0.45;
  const t = Math.min(100, Math.max(0, v));
  return 0.28 + (t / 100) * 0.34;
}

export async function parseLayoutImageFile(file, options = {}) {
  const mode = options.mode === "blueprint" ? "blueprint" : "colormap";
  const image = await createImageBitmap(file);
  const baseName = file.name.replace(/\.[^.]+$/, "") || "Uploaded Layout";
  return parseLayoutFromImageBitmap(image, file.name, baseName, options, mode);
}

function parseLayoutFromImageBitmap(image, fileName, layoutName, options, mode) {
  const gridWidth = Number(options.gridWidth || 50);
  const gridHeight = Number(options.gridHeight || 50);
  const tolerance = Number(options.tolerance || 54);
  const lineFold = optionsBlueprintLineFold(options);
  const canvas = document.createElement("canvas");
  canvas.width = gridWidth;
  canvas.height = gridHeight;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, gridWidth, gridHeight);
  ctx.drawImage(image, 0, 0, gridWidth, gridHeight);
  const { data } = ctx.getImageData(0, 0, gridWidth, gridHeight);
  const lumas =
    mode === "blueprint" ? sortedLuminancesFromImageData(data) : null;

  const grid = [];
  for (let y = 0; y < gridHeight; y += 1) {
    const row = [];
    for (let x = 0; x < gridWidth; x += 1) {
      const index = (y * gridWidth + x) * 4;
      const red = data[index];
      const green = data[index + 1];
      const blue = data[index + 2];
      const alpha = data[index + 3];
      let tile;
      if (mode === "blueprint") {
        tile = classifyBlueprintCell(
          red,
          green,
          blue,
          alpha,
          lumas,
          tolerance,
          lineFold,
        );
      } else if (alpha < 32) {
        tile = "floor";
      } else {
        tile = nearestTile([red, green, blue], tolerance);
      }
      row.push(tile);
    }
    grid.push(row);
  }

  return {
    id: `layout-${Date.now()}`,
    name: layoutName,
    width: gridWidth,
    height: gridHeight,
    grid,
    layout: extractObjects(grid),
    source: {
      fileName,
      imageWidth: image.width,
      imageHeight: image.height,
      gridWidth,
      gridHeight,
      tolerance,
      scanMode: mode,
      blueprintWallStrength: Number(options?.blueprintLineSensitivity ?? 45),
    },
  };
}

export function drawLayoutPreview(canvas, parsed) {
  if (!canvas || !parsed) return;
  const ctx = canvas.getContext("2d");
  const scale = Math.max(1, Math.floor(Math.min(canvas.width / parsed.width, canvas.height / parsed.height)));
  const offsetX = Math.floor((canvas.width - parsed.width * scale) / 2);
  const offsetY = Math.floor((canvas.height - parsed.height * scale) / 2);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#0b1017";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (let y = 0; y < parsed.height; y += 1) {
    for (let x = 0; x < parsed.width; x += 1) {
      const tile = parsed.grid[y][x];
      ctx.fillStyle = TILE_COLORS[tile]?.hex || "#FFFFFF";
      ctx.fillRect(offsetX + x * scale, offsetY + y * scale, scale, scale);
    }
  }
}

function nearestTile(rgb, tolerance) {
  let best = { tile: "floor", distance: Infinity };
  for (const [tile, spec] of Object.entries(TILE_COLORS)) {
    const distance = colorDistance(rgb, spec.rgb);
    if (distance < best.distance) best = { tile, distance };
  }
  return best.distance <= tolerance ? best.tile : "floor";
}

function colorDistance(a, b) {
  return Math.sqrt(
    (a[0] - b[0]) ** 2 +
    (a[1] - b[1]) ** 2 +
    (a[2] - b[2]) ** 2,
  );
}

function extractObjects(grid) {
  const height = grid.length;
  const width = grid[0]?.length || 0;
  const visited = Array.from({ length: height }, () => Array(width).fill(false));
  const layout = {
    shelves: [],
    conveyors: [],
    bins: [],
    chargers: [],
    obstacles: [],
    workerSpawns: [],
  };
  const counters = {
    shelf: 0,
    conveyor: 0,
    bin: 0,
    charger: 0,
    restricted: 0,
    wall: 0,
    worker_spawn: 0,
  };

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const tile = grid[y][x];
      if (visited[y][x] || !OBJECT_TILES.has(tile)) continue;
      const component = floodFill(grid, visited, x, y, tile);
      const object = componentObject(tile, component, counters);
      if (!object) continue;
      if (tile === "shelf") layout.shelves.push(object);
      else if (baseTile(tile) === "conveyor") layout.conveyors.push(object);
      else if (baseTile(tile) === "bin") layout.bins.push(object);
      else if (tile === "charger") layout.chargers.push(object);
      else if (tile === "worker_spawn") layout.workerSpawns.push(object);
      else layout.obstacles.push(object);
    }
  }
  return layout;
}

function floodFill(grid, visited, startX, startY, tile) {
  const stack = [[startX, startY]];
  const cells = [];
  visited[startY][startX] = true;
  while (stack.length) {
    const [x, y] = stack.pop();
    cells.push([x, y]);
    for (const [nx, ny] of [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]]) {
      if (ny < 0 || ny >= grid.length || nx < 0 || nx >= grid[0].length) continue;
      if (visited[ny][nx] || grid[ny][nx] !== tile) continue;
      visited[ny][nx] = true;
      stack.push([nx, ny]);
    }
  }
  return cells;
}

function componentObject(tile, cells, counters) {
  const kind = baseTile(tile);
  const xs = cells.map((cell) => cell[0]);
  const ys = cells.map((cell) => cell[1]);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const maxX = Math.max(...xs);
  const maxY = Math.max(...ys);
  counters[kind] = (counters[kind] || 0) + 1;
  const base = {
    id: `${kind}-${counters[kind]}`,
    tile,
    x: minX,
    y: minY,
    width: maxX - minX + 1,
    depth: maxY - minY + 1,
    cells,
  };
  const packageColor = TILE_COLORS[tile]?.packageColor;
  if (kind === "conveyor") {
    return { ...base, packageColor, spawnIntervalMs: 2500, enabled: true };
  }
  if (kind === "bin") {
    return { ...base, packageColor, acceptsType: packageColor || "auto" };
  }
  return base;
}

function baseTile(tile) {
  if (tile.startsWith("conveyor_")) return "conveyor";
  if (tile.startsWith("bin_")) return "bin";
  return tile;
}
