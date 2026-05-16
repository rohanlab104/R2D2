export function buildRecommendations(metrics, world) {
  const recommendations = [];
  const pending = Number(metrics.pendingTasks || 0);
  const utilization = Number(metrics.workerUtilization || 0);
  const congestion = Number(metrics.congestionScore || 0);
  const average = Number(metrics.averageDeliveryTime || 0);

  if (pending >= 6 && utilization >= 75) {
    recommendations.push("Add one or two workers; backlog is high and the fleet is already busy.");
  }
  if (pending >= 6 && utilization < 55) {
    recommendations.push("Move bins closer to conveyors or remove blocked aisle cells; workers are not staying busy despite backlog.");
  }
  if (congestion >= 25) {
    const zone = metrics.bottleneckZones?.[0];
    recommendations.push(zone
      ? `Widen or reroute the aisle around (${zone.x}, ${zone.y}); several worker paths overlap there.`
      : "Widen the busiest aisle or move shelves away from the main route.");
  }
  if (average >= 12) {
    recommendations.push("Move at least one drop box closer to the intake conveyors to reduce average delivery time.");
  }
  if ((world.chargers || []).length === 0 && (world.workers || []).length > 6) {
    recommendations.push("Add or mark charger zones for larger fleets so the demo can show resilience planning.");
  }
  if (!recommendations.length) {
    recommendations.push("Layout is stable for this run. Try increasing order volume or removing a bin to stress-test bottlenecks.");
  }
  return recommendations.slice(0, 4);
}
