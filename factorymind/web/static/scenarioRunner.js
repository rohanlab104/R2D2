export function createScenario({ name, layoutId, workerCount, orderVolume, metrics }) {
  return {
    scenarioName: name || `Scenario ${new Date().toLocaleTimeString()}`,
    layoutId: layoutId || "live-layout",
    workerCount,
    orderVolume,
    metrics,
    capturedAt: Date.now(),
  };
}

export function compareScenarios(scenarios) {
  if (!scenarios.length) return [];
  const baseline = scenarios[0];
  return scenarios.map((scenario) => ({
    ...scenario,
    deltaThroughput: scenario.metrics.throughput - baseline.metrics.throughput,
    deltaAverageDeliveryTime: scenario.metrics.averageDeliveryTime - baseline.metrics.averageDeliveryTime,
    deltaCompletedTasks: scenario.metrics.completedTasks - baseline.metrics.completedTasks,
    deltaCongestionScore: scenario.metrics.congestionScore - baseline.metrics.congestionScore,
  }));
}
