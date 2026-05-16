export function collectMetrics(world, options = {}) {
  const factories = world.factories || [];
  const workers = world.workers || [];
  const dropboxes = world.dropboxes || [];
  const enabledWorkers = workers.filter((worker) => worker.status !== "disabled");
  const busyWorkers = enabledWorkers.filter((worker) => worker.status && worker.status !== "idle");
  const pendingTasks = factories.reduce((total, factory) => (
    total +
    (factory.conveyor || []).length +
    (factory.pad_packages || []).filter((pkg) => pkg.reserved_by == null).length
  ), 0);
  const completedTasks = dropboxes.reduce((total, box) => total + Number(box.delivered || 0), 0);
  const workerUtilization = enabledWorkers.length
    ? Math.round((busyWorkers.length / enabledWorkers.length) * 100)
    : 0;
  const congestionScore = estimateCongestion(world);
  return {
    throughput: Number(world.stats?.rate || 0),
    averageDeliveryTime: Number(world.stats?.avg_route_time || 0),
    completedTasks,
    pendingTasks,
    workerUtilization,
    congestionScore,
    workerCount: Number(options.workerCount || enabledWorkers.length || 0),
    orderVolume: Number(options.orderVolume || world.order_volume || 0),
    bottleneckZones: findBottleneckZones(world),
  };
}

export function formatMetric(value, suffix = "") {
  if (value == null || Number.isNaN(Number(value))) return "--";
  return `${Number(value).toFixed(Number(value) % 1 ? 1 : 0)}${suffix}`;
}

function estimateCongestion(world) {
  const workers = world.workers || [];
  const blocked = workers.filter((worker) => worker.blocked_by_worker_id != null).length;
  const waitTicks = workers.reduce((total, worker) => total + Number(worker.traffic_wait_ticks || 0), 0);
  const pathDensity = findBottleneckZones(world)[0]?.count || 0;
  return Math.round(blocked * 18 + waitTicks * 3 + Math.max(0, pathDensity - 2) * 5);
}

function findBottleneckZones(world) {
  const counts = new Map();
  for (const worker of world.workers || []) {
    for (const cell of (worker.path || []).slice(0, 10)) {
      const key = `${cell[0]},${cell[1]}`;
      counts.set(key, (counts.get(key) || 0) + 1);
    }
  }
  return [...counts.entries()]
    .map(([key, count]) => {
      const [x, y] = key.split(",").map(Number);
      return { x, y, count };
    })
    .filter((zone) => zone.count > 1)
    .sort((a, b) => b.count - a.count)
    .slice(0, 5);
}
