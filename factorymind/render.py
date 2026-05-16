"""Pygame renderer for the FactoryMind simulation.

Person B owns this file. It is designed to be callable with mock data
independently of the rest of the system.

Coordinate system
-----------------
Each grid cell is CELL_SIZE pixels.
The grid is drawn starting at GRID_OFFSET_X, GRID_OFFSET_Y.
The side panel starts at PANEL_X.
"""

from __future__ import annotations

import sys
import time
from typing import Optional

try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
CELL_SIZE = 16
GRID_OFFSET_X = 50
GRID_OFFSET_Y = 50
PANEL_X = 900

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_BG         = (10,  10,  15)   # #0a0a0f
C_GRID       = (30,  30,  40)
C_ACCENT     = (0,   212, 255)  # #00d4ff
C_TEXT       = (224, 224, 224)  # #e0e0e0
C_LEADER     = (0,   212, 255)  # cyan
C_WORKER     = (255, 140, 0)    # orange
C_WALL       = (80,  80,  100)
C_PANEL_BG   = (15,  15,  22)
C_BTN_DISC   = (200, 30,  30)
C_BTN_OPEN   = (30,  100, 200)
C_BTN_BOTTLE = (100, 30,  200)
C_BANNER     = (180, 20,  20)
C_WHITE      = (255, 255, 255)

# Button rectangles (x, y, w, h) — defined in panel space
_BTN_DISCONNECT  = pygame.Rect(910, 820, 180, 40) if pygame else None
_BTN_OPEN        = pygame.Rect(910, 760, 80,  40) if pygame else None
_BTN_BOTTLENECK  = pygame.Rect(1000, 760, 120, 40) if pygame else None


def _ensure_pygame() -> None:
    """Raise ImportError if pygame is not installed."""
    if pygame is None:
        raise ImportError("pygame is required for rendering. Run: pip install pygame")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_display(width: int = 1400, height: int = 900) -> "pygame.Surface":
    """Initialise pygame and return the main screen surface."""
    _ensure_pygame()
    pygame.init()
    pygame.display.set_caption("FactoryMind — R2D2")
    screen = pygame.display.set_mode((width, height))
    return screen


def render(screen: "pygame.Surface", world_state: dict) -> None:
    """Draw one complete frame from world_state onto screen.

    Calls draw_* helpers and then flips the display.
    """
    _ensure_pygame()
    screen.fill(C_BG)

    wall_set = {tuple(c) for c in world_state.get("wall", [])}

    draw_grid(screen)
    draw_walls(screen, wall_set)
    draw_workstations(screen, world_state.get("workstations", []))
    draw_robots(screen, world_state.get("robots", []))
    draw_sidepanel(screen, world_state)

    if world_state.get("connection_status") == "offline":
        draw_disconnect_banner(screen)

    pygame.display.flip()


def draw_grid(screen: "pygame.Surface") -> None:
    """Draw a faint grid over the simulation area."""
    from factorymind.state import GRID_WIDTH, GRID_HEIGHT
    for x in range(GRID_WIDTH + 1):
        px = GRID_OFFSET_X + x * CELL_SIZE
        pygame.draw.line(screen, C_GRID, (px, GRID_OFFSET_Y),
                         (px, GRID_OFFSET_Y + GRID_HEIGHT * CELL_SIZE))
    for y in range(GRID_HEIGHT + 1):
        py = GRID_OFFSET_Y + y * CELL_SIZE
        pygame.draw.line(screen, C_GRID, (GRID_OFFSET_X, py),
                         (GRID_OFFSET_X + GRID_WIDTH * CELL_SIZE, py))


def draw_walls(screen: "pygame.Surface", wall_set: set) -> None:
    """Draw blocked cells as solid rectangles."""
    for (wx, wy) in wall_set:
        rect = pygame.Rect(
            GRID_OFFSET_X + wx * CELL_SIZE,
            GRID_OFFSET_Y + wy * CELL_SIZE,
            CELL_SIZE, CELL_SIZE,
        )
        pygame.draw.rect(screen, C_WALL, rect)


def draw_workstations(screen: "pygame.Surface", workstations: list[dict]) -> None:
    """Draw each workstation as a coloured square with a label."""
    font = pygame.font.SysFont("monospace", 11)
    for ws in workstations:
        x, y = ws["pos"]
        color = tuple(ws["color"])
        rect = pygame.Rect(
            GRID_OFFSET_X + x * CELL_SIZE - CELL_SIZE,
            GRID_OFFSET_Y + y * CELL_SIZE - CELL_SIZE,
            CELL_SIZE * 3, CELL_SIZE * 3,
        )
        pygame.draw.rect(screen, color, rect, border_radius=4)
        label = font.render(ws["name"], True, C_WHITE)
        screen.blit(label, (rect.x, rect.y - 14))


def draw_robots(screen: "pygame.Surface", robots: list[dict]) -> None:
    """Draw each robot as a small circle coloured by role."""
    from factorymind.state import LEADER
    for robot in robots:
        x, y = robot["pos"]
        cx = GRID_OFFSET_X + x * CELL_SIZE + CELL_SIZE // 2
        cy = GRID_OFFSET_Y + y * CELL_SIZE + CELL_SIZE // 2
        color = C_LEADER if robot["role"] == LEADER else C_WORKER
        pygame.draw.circle(screen, color, (cx, cy), CELL_SIZE // 2 - 1)
        # Draw robot ID
        font = pygame.font.SysFont("monospace", 9)
        label = font.render(str(robot["id"]), True, C_BG)
        screen.blit(label, (cx - 4, cy - 5))


def draw_sidepanel(screen: "pygame.Surface", world_state: dict) -> None:
    """Draw the side panel with stats, blackboard feed, and buttons."""
    global _BTN_DISCONNECT, _BTN_OPEN, _BTN_BOTTLENECK
    panel_rect = pygame.Rect(PANEL_X, 0, screen.get_width() - PANEL_X, screen.get_height())
    pygame.draw.rect(screen, C_PANEL_BG, panel_rect)

    font_title = pygame.font.SysFont("monospace", 18, bold=True)
    font_body  = pygame.font.SysFont("monospace", 12)
    font_small = pygame.font.SysFont("monospace", 10)

    y_cursor = 20

    # Title
    title = font_title.render("FactoryMind R2D2", True, C_ACCENT)
    screen.blit(title, (PANEL_X + 10, y_cursor))
    y_cursor += 30

    # Layout
    layout_label = font_body.render(f"Layout: {world_state.get('layout', '?')}", True, C_TEXT)
    screen.blit(layout_label, (PANEL_X + 10, y_cursor))
    y_cursor += 20

    # Tick
    tick_label = font_body.render(f"Tick: {world_state.get('tick', 0)}", True, C_TEXT)
    screen.blit(tick_label, (PANEL_X + 10, y_cursor))
    y_cursor += 25

    # Stats
    stats = world_state.get("stats", {})
    for key, label in [("completed", "Completed"), ("elapsed", "Elapsed(s)"), ("rate", "Rate/min")]:
        val = stats.get(key, 0)
        text = f"{label}: {val:.1f}" if isinstance(val, float) else f"{label}: {val}"
        surf = font_body.render(text, True, C_TEXT)
        screen.blit(surf, (PANEL_X + 10, y_cursor))
        y_cursor += 18

    y_cursor += 10
    pygame.draw.line(screen, C_ACCENT, (PANEL_X + 5, y_cursor), (screen.get_width() - 5, y_cursor))
    y_cursor += 10

    # Blackboard feed (last 15 messages)
    bb_header = font_body.render("Blackboard (recent):", True, C_ACCENT)
    screen.blit(bb_header, (PANEL_X + 10, y_cursor))
    y_cursor += 18

    messages = world_state.get("blackboard", [])[-15:]
    for msg in reversed(messages):
        line = f"[{msg.get('type','?')}] R{msg.get('from','?')}: {str(msg.get('content',''))[:30]}"
        surf = font_small.render(line, True, C_TEXT)
        screen.blit(surf, (PANEL_X + 10, y_cursor))
        y_cursor += 13
        if y_cursor > 700:
            break

    # Layout switch buttons
    _BTN_OPEN       = pygame.Rect(PANEL_X + 10, 760, 80, 40)
    _BTN_BOTTLENECK = pygame.Rect(PANEL_X + 100, 760, 120, 40)
    _BTN_DISCONNECT = pygame.Rect(PANEL_X + 10, 820, 180, 40)

    pygame.draw.rect(screen, C_BTN_OPEN, _BTN_OPEN, border_radius=6)
    screen.blit(font_body.render("Open", True, C_WHITE), (_BTN_OPEN.x + 10, _BTN_OPEN.y + 10))

    pygame.draw.rect(screen, C_BTN_BOTTLE, _BTN_BOTTLENECK, border_radius=6)
    screen.blit(font_body.render("Bottleneck", True, C_WHITE), (_BTN_BOTTLENECK.x + 5, _BTN_BOTTLENECK.y + 10))

    pygame.draw.rect(screen, C_BTN_DISC, _BTN_DISCONNECT, border_radius=6)
    disc_label = font_body.render("DISCONNECT", True, C_WHITE)
    screen.blit(disc_label, (_BTN_DISCONNECT.x + 20, _BTN_DISCONNECT.y + 10))


def draw_disconnect_banner(screen: "pygame.Surface") -> None:
    """Draw a red banner at the top indicating cloud disconnect."""
    banner_rect = pygame.Rect(0, 0, screen.get_width(), 30)
    pygame.draw.rect(screen, C_BANNER, banner_rect)
    font = pygame.font.SysFont("monospace", 14, bold=True)
    msg = "CLOUD DISCONNECTED — RUNNING LOCALLY ON GX10"
    surf = font.render(msg, True, C_WHITE)
    screen.blit(surf, (screen.get_width() // 2 - surf.get_width() // 2, 6))


def get_button_click(pos: tuple[int, int]) -> Optional[str]:
    """Return which button was clicked, or None.

    Returns "disconnect", "layout_open", "layout_bottleneck", or None.
    """
    if _BTN_DISCONNECT and _BTN_DISCONNECT.collidepoint(pos):
        return "disconnect"
    if _BTN_OPEN and _BTN_OPEN.collidepoint(pos):
        return "layout_open"
    if _BTN_BOTTLENECK and _BTN_BOTTLENECK.collidepoint(pos):
        return "layout_bottleneck"
    return None


# ---------------------------------------------------------------------------
# Standalone smoke test for Person B
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time as _time
    from factorymind.state import create_initial_state, OPEN_FLOOR, LEADER, WORKER

    _ensure_pygame()

    fake_state = create_initial_state(OPEN_FLOOR)
    fake_state["blackboard"] = [
        {"from": 0, "type": "CLAIM",    "content": "task-1", "timestamp": 0.0},
        {"from": 1, "type": "INTENT",   "content": "Assembly", "timestamp": 0.1},
        {"from": 0, "type": "COMPLETE", "content": "task-1", "timestamp": 0.5},
    ]
    fake_state["stats"] = {"completed": 3, "elapsed": 12.0, "rate": 15.0}

    screen = init_display()
    clock = pygame.time.Clock()
    deadline = _time.time() + 5.0

    while _time.time() < deadline:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        render(screen, fake_state)
        clock.tick(60)

    pygame.quit()
    print("render.py smoke test OK — window ran for 5 seconds")
