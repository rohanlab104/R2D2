"""Graphics-free grid pathfinding helpers for warehouse simulations."""

from __future__ import annotations

import heapq

from factorymind.state import GRID_HEIGHT, GRID_WIDTH


def a_star(
    start: list[int],
    goal: list[int],
    wall_set: set[tuple[int, int]],
    *,
    grid_width: int = GRID_WIDTH,
    grid_height: int = GRID_HEIGHT,
) -> list[list[int]]:
    """Return shortest path cells from start exclusive to goal inclusive."""
    sx, sy = start
    gx, gy = goal

    if [sx, sy] == [gx, gy]:
        return []

    open_heap: list[tuple[int, int, int, tuple[int, int]]] = []
    heapq.heappush(open_heap, (0, 0, id((sx, sy)), (sx, sy)))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {(sx, sy): None}
    g_score: dict[tuple[int, int], int] = {(sx, sy): 0}

    def h(x: int, y: int) -> int:
        return abs(x - gx) + abs(y - gy)

    while open_heap:
        _, g, _, current = heapq.heappop(open_heap)
        cx, cy = current

        if [cx, cy] == [gx, gy]:
            path: list[list[int]] = []
            node: tuple[int, int] | None = (cx, cy)
            while node is not None:
                path.append([node[0], node[1]])
                node = came_from[node]
            path.reverse()
            return path[1:]

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < grid_width and 0 <= ny < grid_height):
                continue
            if (nx, ny) in wall_set:
                continue
            ng = g + 1
            if ng < g_score.get((nx, ny), 10**9):
                g_score[(nx, ny)] = ng
                came_from[(nx, ny)] = (cx, cy)
                heapq.heappush(open_heap, (ng + h(nx, ny), ng, id((nx, ny)), (nx, ny)))

    return []


def route_cost(start: list[int], goal: list[int], wall_set: set[tuple[int, int]]) -> int:
    """Return path length, or a large cost when no route exists."""
    if start == goal:
        return 0
    path = a_star(start, goal, wall_set)
    return len(path) if path else 10**6
