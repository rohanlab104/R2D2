export function validateLayout(parsed) {
  const errors = [];
  const warnings = [];
  if (!parsed || !Array.isArray(parsed.grid) || !parsed.grid.length) {
    return { valid: false, errors: ["No parsed grid was produced."], warnings };
  }
  const counts = countTiles(parsed.grid);
  const layout = parsed.layout || {};
  if (!counts.floor) errors.push("Layout must include walkable white floor cells.");
  if (!layout.conveyors?.length) errors.push("Add at least one conveyor/intake spawn point.");
  if (!layout.bins?.length) errors.push("Add at least one bin/dropoff zone.");
  if (!layout.workerSpawns?.length) warnings.push("No yellow worker spawn found; workers will start near the bottom center.");
  if ((layout.conveyors?.length || 0) > (layout.bins?.length || 0) * 3) {
    warnings.push("Many more conveyors than bins; pending tasks may rise quickly.");
  }
  if ((counts.wall || 0) + (counts.shelf || 0) + (counts.restricted || 0) > counts.floor * 1.7) {
    warnings.push("High blocked-area ratio; validate that aisles are connected.");
  }
  return { valid: errors.length === 0, errors, warnings, counts };
}

export function countTiles(grid) {
  const counts = {};
  for (const row of grid) {
    for (const tile of row) {
      counts[tile] = (counts[tile] || 0) + 1;
      const base = baseTile(tile);
      if (base !== tile) counts[base] = (counts[base] || 0) + 1;
    }
  }
  return counts;
}

function baseTile(tile) {
  if (tile.startsWith("conveyor_")) return "conveyor";
  if (tile.startsWith("bin_")) return "bin";
  return tile;
}
