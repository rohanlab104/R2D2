"""Pygame renderer for FactoryMind R2D2 — interactive command console v3.

Person B owns this file.

v3 changes vs v2:
- Side panel is now an interactive command console
- Chat input dispatches task prompts to agents
- Builder mode: click/drag to draw walls, right-click to erase
- 4-tile stats bar adds LAST ROUTE TIME
- Log feed has per-entry opacity decay (newest = full, older = faded)
- CUSTOM layout button (auto-selected when player draws walls)
- Mode toggle: CURSOR / BUILDER
- Primary API is now handle_event(event, world_state) — returns str, dict, or None
- get_button_click() kept for backward compatibility with Person D's main.py
"""

from __future__ import annotations

import math
import sys
import time
from collections import deque
from typing import Optional, Union

try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
CELL_SIZE     = 16
GRID_OFFSET_X = 50
GRID_OFFSET_Y = 50
PANEL_X       = 900
WINDOW_W      = 1400
WINDOW_H      = 900

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_BG          = (10,  10,  20)
C_GRID_FAINT  = (26,  26,  42)
C_GRID_MAJOR  = (37,  37,  64)
C_ACCENT      = (0,   212, 255)
C_TEXT        = (224, 224, 224)
C_DIM         = (100, 100, 130)
C_LEADER      = (0,   212, 255)
C_WORKER      = (255, 140, 0)
C_WALL        = (55,  55,  85)
C_PANEL_BG    = (12,  12,  24)
C_PANEL_EDGE  = (28,  28,  55)
C_WHITE       = (255, 255, 255)
C_FLASH       = (200, 0,   0)
C_BADGE_OK    = (10,  80,  30)

# Message / log entry colours
C_MSG: dict[str, tuple[int, int, int]] = {
    # Original blackboard types
    "CLAIM":      (0,   212, 255),
    "STRATEGY":   (220, 0,   255),
    "BOTTLENECK": (255, 200, 0),
    "COMPLETE":   (0,   255, 136),
    "INTENT":     (180, 180, 210),
    # New agent reasoning types
    "REASONING":  (220, 0,   255),   # magenta
    "FOUND":      (0,   212, 255),   # cyan
    "ERROR":      (255, 200, 0),     # yellow
    "RETRY":      (255, 140, 0),     # orange
    "USER":       (255, 255, 255),   # white
    # NemoClaw / OpenClaw policy events.
    "POLICY":     (0,   255, 200),   # mint — visible against dark UI
}
C_MSG_DEFAULT = (140, 140, 160)

# ---------------------------------------------------------------------------
# Module-level animation / interaction state
# ---------------------------------------------------------------------------

# Robot interpolation (v2)
_robot_visual_pos: dict[int, list[float]] = {}
_robot_trails:     dict[int, deque]       = {}
_TRAIL_LEN = 8
_LERP      = 0.22

# Disconnect flash (v2)
_prev_connection_status: str = "online"
_disconnect_flash_ticks: int = 0
_FLASH_DURATION_MS       = 500

# Interaction mode
_mode: str = "cursor"  # "cursor" | "builder"

# Builder state
_builder_hover_cell:    Optional[tuple[int, int]] = None
_builder_drag_start:    Optional[tuple[int, int]] = None
_builder_dragged_cells: set                       = set()

# Chat input state
_chat_text:    str  = ""
_chat_focused: bool = False
_CHAT_MAX_CHARS = 320

# Button rects — updated every frame by draw_sidepanel
_BTN_DISCONNECT:   Optional["pygame.Rect"] = None
_BTN_MODE_CURSOR:  Optional["pygame.Rect"] = None
_BTN_MODE_BUILDER: Optional["pygame.Rect"] = None
_BTN_RESET:        Optional["pygame.Rect"] = None
_BTN_SPEED:        Optional["pygame.Rect"] = None
_BTN_SEND:         Optional["pygame.Rect"] = None
_BTN_CHAT_INPUT:   Optional["pygame.Rect"] = None


def _ensure_pygame() -> None:
    """Raise ImportError if pygame is not installed."""
    if pygame is None:
        raise ImportError("pygame is required. Run: pip3 install pygame")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _grid_cell_at(pos: tuple[int, int]) -> Optional[tuple[int, int]]:
    """Convert screen (px, py) to grid (gx, gy), or None if outside the grid."""
    from factorymind.state import GRID_WIDTH, GRID_HEIGHT
    px, py = pos
    gx = (px - GRID_OFFSET_X) // CELL_SIZE
    gy = (py - GRID_OFFSET_Y) // CELL_SIZE
    if 0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT:
        return (gx, gy)
    return None


def _line_cells(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Bresenham line — all grid cells from (x0,y0) to (x1,y1) inclusive."""
    cells: list[tuple[int, int]] = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x   += sx
        if e2 < dx:
            err += dx
            y   += sy
    return cells


def _dim(color: tuple[int, int, int], opacity: int) -> tuple[int, int, int]:
    """Scale an RGB colour by opacity (0–255) without alpha blending."""
    return tuple(int(c * opacity // 255) for c in color)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Alpha drawing helper
# ---------------------------------------------------------------------------

def _draw_alpha_circle(
    surface: "pygame.Surface",
    color:   tuple[int, int, int],
    center:  tuple[int, int],
    radius:  int,
    alpha:   int,
) -> None:
    """Blit a per-pixel-alpha circle onto surface."""
    size = radius * 2 + 2
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.circle(s, (*color, alpha), (radius + 1, radius + 1), radius)
    surface.blit(s, (center[0] - radius - 1, center[1] - radius - 1))


# ---------------------------------------------------------------------------
# Public API — init
# ---------------------------------------------------------------------------

def init_display(width: int = WINDOW_W, height: int = WINDOW_H) -> "pygame.Surface":
    """Initialise pygame and return the main screen surface."""
    _ensure_pygame()
    pygame.init()
    pygame.display.set_caption("FactoryMind R2D2 — Hack-a-Claw x NVIDIA")
    return pygame.display.set_mode((width, height))


# ---------------------------------------------------------------------------
# Public API — render
# ---------------------------------------------------------------------------

def render(screen: "pygame.Surface", world_state: dict) -> None:
    """Draw one complete frame from world_state and flip the display."""
    global _prev_connection_status, _disconnect_flash_ticks
    _ensure_pygame()

    # Detect connection_status transitions to trigger flash
    current = world_state.get("connection_status", "online")
    if current != _prev_connection_status:
        _disconnect_flash_ticks = pygame.time.get_ticks()
        _prev_connection_status = current

    # Hide OS cursor in builder mode so the custom cursor takes over
    pygame.mouse.set_visible(_mode != "builder")

    screen.fill(C_BG)
    wall_set = {tuple(c) for c in world_state.get("wall", [])}

    draw_factory_apron(screen)                       # margin scenery (under everything)
    draw_grid(screen)
    draw_factory_floor_extras(screen)                # aisles + extra designated zones (visual)
    draw_factory_interior_structures(screen, wall_set)  # buildings / machinery under walls
    draw_designated_zones(screen, world_state)       # colored floor patches per WS
    draw_charging_area(screen, world_state)          # central dispatch / charging zone
    draw_walls(screen, wall_set)
    draw_builder_overlay(screen, world_state)        # preview sits above walls, below robots
    draw_spawn_points(screen, world_state)
    draw_workstations(screen, world_state.get("workstations", []))
    draw_robots(screen, world_state.get("robots", []))
    draw_sidepanel(screen, world_state)

    if current == "offline":
        draw_disconnect_banner(screen)

    _draw_flash_overlay(screen)
    pygame.display.flip()


# ---------------------------------------------------------------------------
# Public API — event handling
# ---------------------------------------------------------------------------

def handle_event(
    event:       "pygame.event.Event",
    world_state: dict,
) -> "Union[dict, str, None]":
    """Process one pygame event and return an action, or None.

    Call this for every event in the main loop instead of get_button_click.

    Return values
    -------------
    str  : "disconnect", "layout_open", "layout_bottleneck", "layout_custom",
           "mode_cursor", "mode_builder", "reset", "speedup"
    dict : {"type": "user_prompt", "text": str}
           {"type": "wall_add",    "cell": [x, y]}
           {"type": "wall_remove", "cell": [x, y]}
    None : no action this event
    """
    global _builder_drag_start, _builder_dragged_cells, _builder_hover_cell
    global _chat_focused

    if event.type == pygame.MOUSEBUTTONDOWN:
        pos    = event.pos
        button = event.button

        if button == 1:
            # Chat input focus management
            if _BTN_CHAT_INPUT and _BTN_CHAT_INPUT.collidepoint(pos):
                _chat_focused = True
                return None
            if _BTN_SEND and _BTN_SEND.collidepoint(pos):
                return _send_chat()
            # Clicking anywhere else loses chat focus
            _chat_focused = False

            # Panel buttons
            action = _check_panel_buttons(pos)
            if action is not None:
                return action

            # Builder: start drag / add wall cell
            if _mode == "builder":
                cell = _grid_cell_at(pos)
                if cell:
                    _builder_drag_start    = cell
                    _builder_dragged_cells = {cell}
                    return {"type": "wall_add", "cell": list(cell)}

        elif button == 3:  # right-click removes wall
            if _mode == "builder":
                cell = _grid_cell_at(pos)
                if cell:
                    return {"type": "wall_remove", "cell": list(cell)}

    elif event.type == pygame.MOUSEBUTTONUP:
        if event.button == 1:
            _builder_drag_start    = None
            _builder_dragged_cells = set()

    elif event.type == pygame.MOUSEMOTION:
        if _mode == "builder":
            _builder_hover_cell = _grid_cell_at(event.pos)
            # Drag-to-paint: emit wall_add for each newly entered cell
            if _builder_drag_start and pygame.mouse.get_pressed()[0]:
                cell = _grid_cell_at(event.pos)
                if cell and cell not in _builder_dragged_cells:
                    _builder_dragged_cells.add(cell)
                    return {"type": "wall_add", "cell": list(cell)}

    elif event.type == pygame.KEYDOWN:
        if _chat_focused:
            return _handle_chat_keydown(event)

    return None


def get_button_click(pos: tuple[int, int]) -> Optional[str]:
    """Backward-compatible mouse handler — returns a string action or None.

    Prefer handle_event() for new code; this does not handle keyboard,
    builder drag, or chat input.
    """
    return _check_panel_buttons(pos)


# ---------------------------------------------------------------------------
# Internal event helpers
# ---------------------------------------------------------------------------

def _check_panel_buttons(pos: tuple[int, int]) -> Optional[str]:
    """Map a click position to a string action, or None."""
    global _mode
    if _BTN_DISCONNECT   and _BTN_DISCONNECT.collidepoint(pos):   return "disconnect"
    if _BTN_RESET        and _BTN_RESET.collidepoint(pos):        return "reset"
    if _BTN_SPEED        and _BTN_SPEED.collidepoint(pos):        return "speedup"
    if _BTN_MODE_CURSOR  and _BTN_MODE_CURSOR.collidepoint(pos):
        _mode = "cursor"
        if pygame:
            pygame.mouse.set_visible(True)
        return "mode_cursor"
    if _BTN_MODE_BUILDER and _BTN_MODE_BUILDER.collidepoint(pos):
        _mode = "builder"
        if pygame:
            pygame.mouse.set_visible(False)
        return "mode_builder"
    return None


def _handle_chat_keydown(event: "pygame.event.Event") -> Optional[dict]:
    """Process a KEYDOWN while the chat box is focused."""
    global _chat_text, _chat_focused
    if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
        return _send_chat()
    elif event.key == pygame.K_BACKSPACE:
        _chat_text = _chat_text[:-1]
    elif event.key == pygame.K_ESCAPE:
        _chat_focused = False
    elif event.unicode and len(_chat_text) < _CHAT_MAX_CHARS:
        _chat_text += event.unicode
    return None


def _send_chat() -> Optional[dict]:
    """Package current chat text as a user_prompt event and clear the input."""
    global _chat_text, _chat_focused
    text          = _chat_text.strip()
    _chat_text    = ""
    _chat_focused = False
    if text:
        return {"type": "user_prompt", "text": text}
    return None


# ---------------------------------------------------------------------------
# Factory environment — apron, zones, charging area, and decorative props
# ---------------------------------------------------------------------------

# Cached random props so they don't jitter every frame.
_DECOR_CACHE: list[tuple] = []


def _grid_bounds() -> tuple[int, int, int, int]:
    """Return (left, top, right, bottom) pixel bounds of the playable grid."""
    from factorymind.state import GRID_WIDTH, GRID_HEIGHT
    return (
        GRID_OFFSET_X,
        GRID_OFFSET_Y,
        GRID_OFFSET_X + GRID_WIDTH * CELL_SIZE,
        GRID_OFFSET_Y + GRID_HEIGHT * CELL_SIZE,
    )


def _build_decor() -> list[tuple]:
    """Generate a fixed set of decorative props in the margins around the grid.

    Each entry is ``(kind, cx, cy, rot_deg)`` and is drawn purely for atmosphere.
    Props live OUTSIDE the playable area so they never affect pathfinding.
    """
    import random
    rng = random.Random(1337)
    left, top, right, bottom = _grid_bounds()

    # Margin strips between the grid and the side panel / window edges.
    strip_left   = (4, top, left - 8, bottom)            # left of grid
    strip_right  = (right + 4, top, PANEL_X - 4, bottom) # right of grid (before panel)
    strip_top    = (left, 4, right, top - 8)             # above grid
    strip_bottom = (left, bottom + 4, right, WINDOW_H - 4)

    props: list[tuple] = []

    def scatter(strip, count):
        x0, y0, x1, y1 = strip
        if x1 - x0 < 14 or y1 - y0 < 14:
            return
        for _ in range(count):
            cx = rng.randint(x0 + 6, x1 - 6)
            cy = rng.randint(y0 + 6, y1 - 6)
            r  = rng.random()
            kind = "pallet" if r < 0.45 else "drum" if r < 0.8 else "cone"
            props.append((kind, cx, cy, rng.randint(0, 359)))

    scatter(strip_left,   8)
    scatter(strip_right,  6)
    scatter(strip_top,    7)
    scatter(strip_bottom, 9)

    # A pair of "trucks" parked at top-right and bottom-left corners.
    if right + 38 < PANEL_X:
        props.append(("truck", right + 24, bottom - 38, 0))
    if left - 32 > 8:
        props.append(("truck", left - 24, top + 32, 180))

    return props


def _draw_pallet(screen: "pygame.Surface", cx: int, cy: int) -> None:
    """Tiny stacked-crate pallet sprite (~12x10 px)."""
    # Wood pallet base
    pygame.draw.rect(screen, (77, 47, 21),  pygame.Rect(cx - 7, cy + 3, 14, 3))
    pygame.draw.rect(screen, (50, 30, 14),  pygame.Rect(cx - 7, cy + 3, 14, 3), 1)
    # Stacked crates
    pygame.draw.rect(screen, (107, 70, 38), pygame.Rect(cx - 6, cy - 1, 6, 4))
    pygame.draw.rect(screen, (50, 30, 14),  pygame.Rect(cx - 6, cy - 1, 6, 4), 1)
    pygame.draw.rect(screen, (140, 50, 50), pygame.Rect(cx,     cy - 1, 6, 4))
    pygame.draw.rect(screen, (60, 18, 18),  pygame.Rect(cx,     cy - 1, 6, 4), 1)
    # Top crate
    pygame.draw.rect(screen, (107, 70, 38), pygame.Rect(cx - 4, cy - 5, 8, 4))
    pygame.draw.rect(screen, (50, 30, 14),  pygame.Rect(cx - 4, cy - 5, 8, 4), 1)


def _draw_drum(screen: "pygame.Surface", cx: int, cy: int) -> None:
    """Small green oil drum sprite."""
    pygame.draw.ellipse(screen, (15, 60, 15),  pygame.Rect(cx - 4, cy - 5, 8, 10))
    pygame.draw.ellipse(screen, (40, 110, 40), pygame.Rect(cx - 4, cy - 5, 8, 10), 1)
    pygame.draw.line(screen, (8, 35, 8), (cx - 4, cy), (cx + 3, cy), 1)
    pygame.draw.ellipse(screen, (10, 40, 10), pygame.Rect(cx - 4, cy - 5, 8, 3))


def _draw_cone(screen: "pygame.Surface", cx: int, cy: int) -> None:
    """Hazard cone sprite."""
    pygame.draw.polygon(screen, (255, 120, 24),
                        [(cx, cy - 6), (cx + 4, cy + 3), (cx - 4, cy + 3)])
    pygame.draw.polygon(screen, (180, 70, 10),
                        [(cx, cy - 6), (cx + 4, cy + 3), (cx - 4, cy + 3)], 1)
    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(cx - 3, cy - 1, 6, 1))
    pygame.draw.rect(screen, (50, 50, 60),    pygame.Rect(cx - 5, cy + 3, 10, 2))


def _draw_truck(screen: "pygame.Surface", cx: int, cy: int, flipped: bool) -> None:
    """Boxy delivery truck sprite (~46x18 px)."""
    body_color = (200, 175, 60)
    cab_color  = (160, 130, 35)
    if flipped:
        # Trailer on the right, cab on the left (top-left corner truck).
        pygame.draw.rect(screen, body_color, pygame.Rect(cx - 4,  cy - 6, 30, 14))
        pygame.draw.rect(screen, (60, 40, 10), pygame.Rect(cx - 4, cy - 6, 30, 14), 1)
        pygame.draw.rect(screen, cab_color,  pygame.Rect(cx - 18, cy - 5, 14, 13))
        pygame.draw.rect(screen, (60, 40, 10), pygame.Rect(cx - 18, cy - 5, 14, 13), 1)
        # Windshield
        pygame.draw.rect(screen, (140, 200, 240), pygame.Rect(cx - 16, cy - 3, 4, 6))
        # Wheels
        for wx in (cx - 14, cx - 6, cx + 4, cx + 18):
            pygame.draw.circle(screen, (15, 15, 20), (wx, cy + 8), 3)
    else:
        # Trailer on the left, cab on the right (bottom-right corner truck).
        pygame.draw.rect(screen, body_color, pygame.Rect(cx - 26, cy - 6, 30, 14))
        pygame.draw.rect(screen, (60, 40, 10), pygame.Rect(cx - 26, cy - 6, 30, 14), 1)
        pygame.draw.rect(screen, cab_color,  pygame.Rect(cx + 4,  cy - 5, 14, 13))
        pygame.draw.rect(screen, (60, 40, 10), pygame.Rect(cx + 4, cy - 5, 14, 13), 1)
        pygame.draw.rect(screen, (140, 200, 240), pygame.Rect(cx + 12, cy - 3, 4, 6))
        for wx in (cx - 18, cx - 8, cx + 6, cx + 16):
            pygame.draw.circle(screen, (15, 15, 20), (wx, cy + 8), 3)


def draw_factory_apron(screen: "pygame.Surface") -> None:
    """Background scenery in the margins around the grid + decorative props."""
    global _DECOR_CACHE
    left, top, right, bottom = _grid_bounds()

    # Concrete apron bands surrounding the grid.
    apron_color = (16, 16, 28)
    pygame.draw.rect(screen, apron_color, pygame.Rect(0, 0, WINDOW_W - (WINDOW_W - PANEL_X), top))
    pygame.draw.rect(screen, apron_color, pygame.Rect(0, bottom, PANEL_X, WINDOW_H - bottom))
    pygame.draw.rect(screen, apron_color, pygame.Rect(0, top, left, bottom - top))
    pygame.draw.rect(screen, apron_color, pygame.Rect(right, top, PANEL_X - right, bottom - top))

    # Painted shop-floor border highlight along the grid edge.
    pygame.draw.rect(screen, (28, 28, 50),
                     pygame.Rect(left - 2, top - 2, right - left + 4, bottom - top + 4), 2)

    # Hazard chevrons in each corner of the playable floor.
    chev_y = (255, 200, 40)
    for (x0, y0, dx, dy) in [
        (left - 1, top - 1,  1,  1),
        (right - 8, top - 1, -1, 1),
        (left - 1, bottom - 8, 1, -1),
        (right - 8, bottom - 8, -1, -1),
    ]:
        for i in range(0, 9, 3):
            pygame.draw.line(screen, chev_y,
                             (x0 + i * dx, y0), (x0, y0 + i * dy), 1)

    # Corner pillars (between the apron and the wall corners).
    for (cx, cy) in [
        (left - 14, top - 14), (right + 14, top - 14),
        (left - 14, bottom + 14), (right + 14, bottom + 14),
    ]:
        if 4 <= cx <= PANEL_X - 4 and 4 <= cy <= WINDOW_H - 4:
            pygame.draw.rect(screen, (42, 42, 68),
                             pygame.Rect(cx - 7, cy - 7, 14, 14))
            pygame.draw.rect(screen, (90, 90, 130),
                             pygame.Rect(cx - 7, cy - 7, 14, 14), 1)
            # Hazard cap
            pygame.draw.rect(screen, (255, 176, 0),
                             pygame.Rect(cx - 8, cy - 8, 16, 3))

    # Cached decorative props.
    if not _DECOR_CACHE:
        _DECOR_CACHE = _build_decor()
    for kind, cx, cy, rot in _DECOR_CACHE:
        if kind == "pallet":
            _draw_pallet(screen, cx, cy)
        elif kind == "drum":
            _draw_drum(screen, cx, cy)
        elif kind == "cone":
            _draw_cone(screen, cx, cy)
        elif kind == "truck":
            _draw_truck(screen, cx, cy, flipped=(rot == 180))


def draw_designated_zones(screen: "pygame.Surface", world_state: dict) -> None:
    """Translucent colored floor patches + hazard chevron borders per workstation."""
    from factorymind.state import GRID_WIDTH, GRID_HEIGHT
    zone_half = 4   # 8x8 cell zone
    for ws in world_state.get("workstations", []):
        x, y  = ws["pos"]
        color = tuple(ws["color"])
        cx_g  = max(zone_half, min(GRID_WIDTH  - zone_half, x))
        cy_g  = max(zone_half, min(GRID_HEIGHT - zone_half, y))
        px = GRID_OFFSET_X + (cx_g - zone_half) * CELL_SIZE
        py = GRID_OFFSET_Y + (cy_g - zone_half) * CELL_SIZE
        size = zone_half * 2 * CELL_SIZE

        # Zone tint
        s = pygame.Surface((size, size), pygame.SRCALPHA)
        s.fill((*color, 36))
        screen.blit(s, (px, py))

        # Hazard chevron border
        border = pygame.Surface((size, size), pygame.SRCALPHA)
        chev = (255, 214, 51, 130)
        pygame.draw.rect(border, chev, pygame.Rect(0, 0, size, 4))
        pygame.draw.rect(border, chev, pygame.Rect(0, size - 4, size, 4))
        pygame.draw.rect(border, chev, pygame.Rect(0, 0, 4, size))
        pygame.draw.rect(border, chev, pygame.Rect(size - 4, 0, 4, size))
        screen.blit(border, (px, py))

        # Zone name painted on the floor
        font = pygame.font.SysFont("monospace", 11, bold=True)
        lbl  = font.render(ws["name"].upper() + " AREA", True, color)
        screen.blit(lbl, (px + 6, py + 6))


def _cell_block_rect(gx0: int, gy0: int, gx1: int, gy1: int) -> "pygame.Rect":
    """Pixel rectangle covering grid cells [gx0,gx1) × [gy0,gy1)."""
    return pygame.Rect(
        GRID_OFFSET_X + gx0 * CELL_SIZE,
        GRID_OFFSET_Y + gy0 * CELL_SIZE,
        (gx1 - gx0) * CELL_SIZE,
        (gy1 - gy0) * CELL_SIZE,
    )


def draw_factory_floor_extras(screen: "pygame.Surface") -> None:
    """Painted shop-floor lanes and extra designated areas (no pathfinding impact)."""
    from factorymind.state import GRID_WIDTH, GRID_HEIGHT
    left, top, right, bottom = _grid_bounds()

    # ── Main forklift aisles (center cross) ───────────────────────────────
    axle = (218, 186, 52)
    vx = GRID_OFFSET_X + (GRID_WIDTH // 2) * CELL_SIZE + CELL_SIZE // 2
    pygame.draw.line(screen, axle, (vx, top), (vx, bottom), 2)
    hy = GRID_OFFSET_Y + (GRID_HEIGHT // 2) * CELL_SIZE + CELL_SIZE // 2
    pygame.draw.line(screen, axle, (left, hy), (right, hy), 2)
    # Dashed lane spine (alternating dark gaps on the vertical)
    for y in range(top + 4, bottom - 2, 12):
        pygame.draw.line(screen, (90, 78, 28), (vx - 1, y), (vx - 1, y + 5), 1)
    for x in range(left + 4, right - 2, 12):
        pygame.draw.line(screen, (90, 78, 28), (x, hy - 1), (x + 5, hy - 1), 1)

    # ── Additional designated floor areas ─────────────────────────────────
    zones = [
        ("STAGING / WIP",     (215, 175, 70),  (255, 200, 75), 29, 29, 41, 41),
        ("BUFFER LANE",       (68,  92, 130), (180, 210, 255), 18, 32, 27, 41),
        ("TOOL CRIB",         (72, 110, 82),  (140, 220, 150),  8, 18, 16, 27),
    ]
    font = pygame.font.SysFont("monospace", 10, bold=True)
    for name, fill_rgb, border_rgb, gx0, gy0, gx1, gy1 in zones:
        r = _cell_block_rect(gx0, gy0, gx1, gy1)
        s = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
        s.fill((*fill_rgb, 28))
        screen.blit(s, r.topleft)
        pygame.draw.rect(screen, border_rgb, r, 2)
        # Hatch hint on two edges
        for i in range(0, r.w, 10):
            pygame.draw.line(screen, _dim(border_rgb, 120),
                             (r.x + i, r.y), (r.x + i + 6, r.y + 6), 1)
        lbl = font.render(name, True, border_rgb)
        screen.blit(lbl, (r.x + 6, r.y + 5))


def _draw_conveyor_belt(screen: "pygame.Surface", gx0: int, gy: int, gx1: int) -> None:
    """Low-profile conveyor strip along a row of cells."""
    r = _cell_block_rect(gx0, gy, gx1, gy + 1)
    pygame.draw.rect(screen, (35, 35, 52), r.inflate(0, -6))
    pygame.draw.rect(screen, (70, 72, 98), r.inflate(0, -6), 1)
    for x in range(r.x + 2, r.right - 2, 7):
        pygame.draw.line(screen, (110, 112, 140), (x, r.y + 4), (x + 4, r.y + 8), 1)


def _draw_machine_cell(screen: "pygame.Surface", gx: int, gy: int,
                       base: tuple[int, int, int],
                       accent: tuple[int, int, int]) -> None:
    """Small CNC / compressor footprint (2×2 cells)."""
    r = _cell_block_rect(gx, gy, gx + 2, gy + 2).inflate(-4, -4)
    pygame.draw.rect(screen, base, r, border_radius=3)
    pygame.draw.rect(screen, accent, r, 2, border_radius=3)
    cx, cy = r.centerx, r.centery
    pygame.draw.circle(screen, accent, (cx, cy), 5)
    pygame.draw.rect(screen, _dim(base, 200), (r.x + 4, r.y + 3, r.w - 8, 4))


def draw_factory_interior_structures(
    screen: "pygame.Surface",
    wall_set: set[tuple[int, int]],
) -> None:
    """Warehouse modules, racks, and floor obstacles — visual only; not in ``wall``."""
    # Low office / subassembly bays (footprints sit behind workstation art).
    modules = [
        (18, 8,  26, 16, (34, 38, 58)),
        (31, 17, 41, 25, (38, 34, 56)),
        (9,  33, 18, 42, (30, 36, 54)),
    ]
    for gx0, gy0, gx1, gy1, fill in modules:
        r = _cell_block_rect(gx0, gy0, gx1, gy1)
        pygame.draw.rect(screen, fill, r)
        hi = tuple(min(255, c + 45) for c in fill)
        pygame.draw.line(screen, hi, (r.x + 3, r.y + 5), (r.right - 4, r.y + 5), 2)
        pygame.draw.rect(screen, _dim(hi, 160), r, 2)
        # Roof louvre lines
        for yy in range(r.y + 12, r.bottom - 6, 9):
            pygame.draw.line(screen, _dim(fill, 90), (r.x + 4, yy), (r.right - 5, yy), 1)

    # Tall rack silhouette along two north–south aisles (thin columns).
    rack_xs = [11, 37]
    for gx in rack_xs:
        for gz in range(6, 44, 3):
            if (gx, gz) in wall_set:
                continue
            px = GRID_OFFSET_X + gx * CELL_SIZE + CELL_SIZE // 2
            py_top = GRID_OFFSET_Y + gz * CELL_SIZE + 2
            pygame.draw.line(screen, (48, 50, 72), (px, py_top),
                             (px, py_top + CELL_SIZE - 6), 3)
            pygame.draw.line(screen, (85, 88, 115), (px - 4, py_top + 3),
                             (px + 4, py_top + 3), 1)

    _draw_conveyor_belt(screen, 20, 14, 29)
    _draw_machine_cell(screen, 33, 9, (40, 42, 62), (0, 212, 255))
    _draw_machine_cell(screen, 14, 37, (48, 44, 58), (255, 140, 60))

    # Floor hazard islands (tape + barrel clusters)
    for (cx, cy) in [(24, 22), (40, 32)]:
        cell = pygame.Rect(
            GRID_OFFSET_X + cx * CELL_SIZE + 3,
            GRID_OFFSET_Y + cy * CELL_SIZE + 3,
            CELL_SIZE - 6, CELL_SIZE - 6,
        )
        pygame.draw.rect(screen, (55, 45, 22), cell, border_radius=2)
        pygame.draw.rect(screen, (210, 170, 40), cell, 1, border_radius=2)
        _draw_drum(screen, cell.centerx, cell.centery - 2)


def draw_charging_area(screen: "pygame.Surface", world_state: dict) -> None:
    """Cyan dispatch/charging zone + dock pixels at each spawn cell."""
    spawns = world_state.get("spawn_positions", [])
    if not spawns:
        return
    xs = [s[0] for s in spawns]
    ys = [s[1] for s in spawns]
    pad_x, pad_y = 3, 3
    minx, maxx = min(xs) - pad_x, max(xs) + pad_x + 1
    miny, maxy = min(ys) - pad_y, max(ys) + pad_y + 1
    px = GRID_OFFSET_X + minx * CELL_SIZE
    py = GRID_OFFSET_Y + miny * CELL_SIZE
    w  = (maxx - minx) * CELL_SIZE
    h  = (maxy - miny) * CELL_SIZE

    # Soft cyan zone fill
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    s.fill((0, 212, 255, 55))
    screen.blit(s, (px, py))

    # Dashed cyan border for the dispatch perimeter
    dash = (0, 212, 255)
    for off in range(0, w, 8):
        pygame.draw.rect(screen, dash, pygame.Rect(px + off, py, 4, 2))
        pygame.draw.rect(screen, dash, pygame.Rect(px + off, py + h - 2, 4, 2))
    for off in range(0, h, 8):
        pygame.draw.rect(screen, dash, pygame.Rect(px, py + off, 2, 4))
        pygame.draw.rect(screen, dash, pygame.Rect(px + w - 2, py + off, 2, 4))

    # Floor label
    font = pygame.font.SysFont("monospace", 10, bold=True)
    lbl  = font.render("CHARGING / DISPATCH", True, (180, 240, 255))
    screen.blit(lbl, (px + 4, py + 3))

    # Charging dock blips at each spawn cell.
    for sx, sy in spawns:
        dx = GRID_OFFSET_X + sx * CELL_SIZE + 2
        dy = GRID_OFFSET_Y + sy * CELL_SIZE + CELL_SIZE - 5
        pygame.draw.rect(screen, (10, 50, 75),  pygame.Rect(dx, dy, 4, 4))
        pygame.draw.rect(screen, (0, 212, 255), pygame.Rect(dx, dy, 4, 1))


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def draw_grid(screen: "pygame.Surface") -> None:
    """Blueprint grid: faint lines with brighter major lines every 5 cells."""
    from factorymind.state import GRID_WIDTH, GRID_HEIGHT
    for x in range(GRID_WIDTH + 1):
        color = C_GRID_MAJOR if x % 5 == 0 else C_GRID_FAINT
        px    = GRID_OFFSET_X + x * CELL_SIZE
        pygame.draw.line(screen, color, (px, GRID_OFFSET_Y),
                         (px, GRID_OFFSET_Y + GRID_HEIGHT * CELL_SIZE))
    for y in range(GRID_HEIGHT + 1):
        color = C_GRID_MAJOR if y % 5 == 0 else C_GRID_FAINT
        py    = GRID_OFFSET_Y + y * CELL_SIZE
        pygame.draw.line(screen, color, (GRID_OFFSET_X, py),
                         (GRID_OFFSET_X + GRID_WIDTH * CELL_SIZE, py))


# ---------------------------------------------------------------------------
# Walls
# ---------------------------------------------------------------------------

def draw_walls(screen: "pygame.Surface", wall_set: set) -> None:
    """Blocked cells as dark filled rectangles with a dim border."""
    for (wx, wy) in wall_set:
        rect = pygame.Rect(
            GRID_OFFSET_X + wx * CELL_SIZE,
            GRID_OFFSET_Y + wy * CELL_SIZE,
            CELL_SIZE, CELL_SIZE,
        )
        pygame.draw.rect(screen, C_WALL, rect)
        pygame.draw.rect(screen, (80, 80, 110), rect, 1)


# ---------------------------------------------------------------------------
# Builder overlay
# ---------------------------------------------------------------------------

def draw_builder_overlay(screen: "pygame.Surface", world_state: dict) -> None:
    """Builder mode: hover cell preview, drag-line preview, custom cursor."""
    if _mode != "builder":
        return

    mouse_pos = pygame.mouse.get_pos()
    hover     = _grid_cell_at(mouse_pos)
    wall_set  = {tuple(c) for c in world_state.get("wall", [])}

    # Faint drag-line preview between drag start and current hover
    if _builder_drag_start and hover and hover != _builder_drag_start:
        sx, sy = _builder_drag_start
        ex, ey = hover
        for cell in _line_cells(sx, sy, ex, ey):
            if cell not in _builder_dragged_cells and cell not in wall_set:
                dx, dy = cell
                s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                s.fill((0, 212, 255, 45))
                screen.blit(s, (GRID_OFFSET_X + dx * CELL_SIZE,
                                GRID_OFFSET_Y + dy * CELL_SIZE))

    # Hover cell highlight
    if hover:
        gx, gy = hover
        px = GRID_OFFSET_X + gx * CELL_SIZE
        py = GRID_OFFSET_Y + gy * CELL_SIZE
        s  = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        s.fill((0, 212, 255, 60 if hover in wall_set else 110))
        screen.blit(s, (px, py))
        pygame.draw.rect(screen, C_ACCENT, (px, py, CELL_SIZE, CELL_SIZE), 1)

    # Custom cursor square follows mouse (OS cursor is hidden in builder mode)
    mx, my = mouse_pos
    cs = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
    cs.fill((0, 212, 255, 50))
    pygame.draw.rect(cs, (0, 212, 255, 200), (0, 0, CELL_SIZE, CELL_SIZE), 2)
    screen.blit(cs, (mx - CELL_SIZE // 2, my - CELL_SIZE // 2))


# ---------------------------------------------------------------------------
# Workstations
# ---------------------------------------------------------------------------

def draw_spawn_points(screen: "pygame.Surface", world_state: dict) -> None:
    """Faint home-base rings at each robot spawn position."""
    for sp in world_state.get("spawn_positions", []):
        x, y = sp
        px = GRID_OFFSET_X + x * CELL_SIZE + CELL_SIZE // 2
        py = GRID_OFFSET_Y + y * CELL_SIZE + CELL_SIZE // 2
        _draw_alpha_circle(screen, C_ACCENT, (px, py), 5, 35)
        pygame.draw.circle(screen, _dim(C_ACCENT, 70), (px, py), 2)


_WS_ABBREV = {"Parts": "P", "Assembly": "A", "QA": "Q", "Shipping": "S"}

def draw_workstations(screen: "pygame.Surface", workstations: list[dict]) -> None:
    """Industrial equipment tiles: dark fill, 3D highlight, bold letter, pill label."""
    font_letter = pygame.font.SysFont("monospace", 28, bold=True)
    font_label  = pygame.font.SysFont("monospace", 11)

    for ws in workstations:
        x, y      = ws["pos"]
        color     = tuple(ws["color"])
        fill      = tuple(max(0, c - 120) for c in color)
        border    = tuple(max(0, c - 80)  for c in color)
        highlight = tuple(min(255, c + 60) for c in color)

        px   = GRID_OFFSET_X + x * CELL_SIZE - CELL_SIZE
        py   = GRID_OFFSET_Y + y * CELL_SIZE - CELL_SIZE
        w    = CELL_SIZE * 3
        h    = CELL_SIZE * 3
        rect = pygame.Rect(px, py, w, h)

        pygame.draw.rect(screen, fill,   rect, border_radius=4)
        pygame.draw.rect(screen, border, rect, 2, border_radius=4)

        hl = pygame.Surface((w - 6, 3), pygame.SRCALPHA)
        hl.fill((*highlight, 110))
        screen.blit(hl, (px + 3, py + 3))

        letter = _WS_ABBREV.get(ws["name"], ws["name"][0])
        ls = font_letter.render(letter, True, color)
        screen.blit(ls, (px + w // 2 - ls.get_width() // 2,
                         py + h // 2 - ls.get_height() // 2))

        lb   = font_label.render(ws["name"].upper(), True, (220, 220, 220))
        pw   = lb.get_width() + 10
        ph   = lb.get_height() + 4
        pill = pygame.Rect(px + w // 2 - pw // 2, py + h + 4, pw, ph)
        pygame.draw.rect(screen, (20, 20, 35), pill, border_radius=3)
        pygame.draw.rect(screen, border, pill, 1, border_radius=3)
        screen.blit(lb, (pill.x + 5, pill.y + 2))


# ---------------------------------------------------------------------------
# Robots + trails
# ---------------------------------------------------------------------------

def draw_robots(screen: "pygame.Surface", robots: list[dict]) -> None:
    """Smooth lerp movement with 8-position fading trails."""
    from factorymind.state import LEADER
    font = pygame.font.SysFont("monospace", 9, bold=True)

    for robot in robots:
        rid    = robot["id"]
        role   = robot["role"]
        gx, gy = robot["pos"]

        if rid not in _robot_visual_pos:
            _robot_visual_pos[rid] = [float(gx), float(gy)]
        if rid not in _robot_trails:
            _robot_trails[rid] = deque(maxlen=_TRAIL_LEN)

        vx, vy = _robot_visual_pos[rid]
        vx += (gx - vx) * _LERP
        vy += (gy - vy) * _LERP
        _robot_visual_pos[rid] = [vx, vy]

        px = int(GRID_OFFSET_X + vx * CELL_SIZE + CELL_SIZE // 2)
        py = int(GRID_OFFSET_Y + vy * CELL_SIZE + CELL_SIZE // 2)
        _robot_trails[rid].append((px, py))

        color = C_LEADER if role == LEADER else C_WORKER
        trail = list(_robot_trails[rid])
        n     = len(trail)

        for i, (tx, ty) in enumerate(trail[:-1]):
            frac   = (i + 1) / n
            _draw_alpha_circle(screen, color, (tx, ty),
                               max(1, int(1 + frac * 4)),
                               int(15 + frac * 120))

        radius = CELL_SIZE // 2 - 1
        _draw_alpha_circle(screen, color, (px, py), radius + 3, 35)
        pygame.draw.circle(screen, color, (px, py), radius)
        pygame.draw.circle(screen, (0, 0, 0), (px, py), radius, 1)

        lbl = font.render(str(rid), True, (0, 0, 0))
        screen.blit(lbl, (px - lbl.get_width() // 2, py - lbl.get_height() // 2))


# ---------------------------------------------------------------------------
# Side panel — interactive command console
# ---------------------------------------------------------------------------

def draw_sidepanel(screen: "pygame.Surface", world_state: dict) -> None:
    """Interactive ops-dashboard: stats, mode/layout controls, agent log, chat."""
    global _BTN_DISCONNECT
    global _BTN_MODE_CURSOR, _BTN_MODE_BUILDER
    global _BTN_RESET, _BTN_SPEED, _BTN_SEND, _BTN_CHAT_INPUT

    W   = screen.get_width() - PANEL_X
    H   = screen.get_height()
    PAD = 14
    x0  = PANEL_X + PAD

    offline = world_state.get("connection_status") == "offline"
    speed   = world_state.get("speed_multiplier", 1)
    active  = world_state.get("layout", "")
    stats   = world_state.get("stats", {})

    pygame.draw.rect(screen, C_PANEL_BG,   (PANEL_X, 0, W, H))
    pygame.draw.line(screen, C_PANEL_EDGE, (PANEL_X, 0), (PANEL_X, H))

    f_title = pygame.font.SysFont("monospace", 20, bold=True)
    f_caps  = pygame.font.SysFont("monospace", 10)
    f_body  = pygame.font.SysFont("monospace", 12)
    f_big   = pygame.font.SysFont("monospace", 19, bold=True)
    f_small = pygame.font.SysFont("monospace", 10)
    f_btn   = pygame.font.SysFont("monospace", 11, bold=True)
    f_chat  = pygame.font.SysFont("monospace", 12)

    y = 14

    # ── Title + underline ──────────────────────────────────────────────────
    ts = f_title.render("FACTORYMIND", True, C_ACCENT)
    screen.blit(ts, (x0, y))
    y += ts.get_height() + 2
    pygame.draw.line(screen, C_ACCENT, (x0, y), (PANEL_X + W - PAD, y), 1)
    y += 8

    # ── 4 stats tiles ─────────────────────────────────────────────────────
    elapsed   = stats.get("elapsed", 0.0)
    last_rt   = stats.get("last_route_time", None)
    avg_rt    = stats.get("avg_route_time",  None)
    last_str  = f"{last_rt:.2f}s" if last_rt is not None else "--"
    avg_str   = f"avg {avg_rt:.1f}s" if avg_rt is not None else ""

    tiles = [
        ("COMPLETED",   str(stats.get("completed", 0)), ""),
        ("TIME",        f"{int(elapsed)//60:02d}:{int(elapsed)%60:02d}", ""),
        ("RATE/MIN",    f"{stats.get('rate', 0.0):.1f}", ""),
        ("LAST ROUTE",  last_str, avg_str),
    ]
    tw = (W - PAD * 2 - 12) // 4
    th = 52
    for i, (lbl, val, sub) in enumerate(tiles):
        tx   = x0 + i * (tw + 4)
        trec = pygame.Rect(tx, y, tw, th)
        pygame.draw.rect(screen, (18, 18, 36), trec, border_radius=4)
        pygame.draw.rect(screen, (40, 40, 70), trec, 1, border_radius=4)
        ls = f_small.render(lbl, True, C_DIM)
        vs = f_big.render(val,   True, C_ACCENT)
        screen.blit(ls, (tx + tw // 2 - ls.get_width() // 2, y + 4))
        screen.blit(vs, (tx + tw // 2 - vs.get_width() // 2, y + 17))
        if sub:
            ss = f_small.render(sub, True, C_DIM)
            screen.blit(ss, (tx + tw // 2 - ss.get_width() // 2, y + 39))
    y += th + 7

    # ── Mode toggle: CURSOR | BUILDER ─────────────────────────────────────
    mbw = (W - PAD * 2 - 8) // 2
    _BTN_MODE_CURSOR  = pygame.Rect(x0,          y, mbw, 26)
    _BTN_MODE_BUILDER = pygame.Rect(x0 + mbw + 8, y, mbw, 26)
    for btn, label, key in [
        (_BTN_MODE_CURSOR,  "✦ CURSOR",  "cursor"),
        (_BTN_MODE_BUILDER, "✏ BUILDER", "builder"),
    ]:
        active_btn = (_mode == key)
        pygame.draw.rect(screen, (15, 25, 50) if active_btn else (12, 15, 28), btn, border_radius=4)
        pygame.draw.rect(screen, C_ACCENT if active_btn else (50, 50, 80),
                         btn, 2 if active_btn else 1, border_radius=4)
        t = f_caps.render(label, True, C_ACCENT if active_btn else C_DIM)
        screen.blit(t, (btn.x + btn.w // 2 - t.get_width() // 2,
                        btn.y + btn.h // 2 - t.get_height() // 2))
    y += 26 + 7

    # ── Layout status pill ────────────────────────────────────────────────
    layout_label = active if active else "OPEN_FLOOR"
    ls_pill = f_caps.render(f"layout: {layout_label}", True, C_DIM)
    screen.blit(ls_pill, (x0, y + 4))
    y += ls_pill.get_height() + 10

    # ── Divider ───────────────────────────────────────────────────────────
    pygame.draw.line(screen, C_PANEL_EDGE, (x0, y), (PANEL_X + W - PAD, y))
    y += 8

    # ── AGENT REASONING header ─────────────────────────────────────────────
    ar = f_body.render("AGENT REASONING", True, C_ACCENT)
    screen.blit(ar, (x0, y))
    tk = f_small.render(f"tick {world_state.get('tick', 0)}  {speed}x", True, C_DIM)
    screen.blit(tk, (PANEL_X + W - PAD - tk.get_width(), y + 2))
    y += ar.get_height() + 5

    # ── Log feed (newest first, opacity decay) ─────────────────────────────
    feed_top    = y
    feed_bottom = H - 188   # leave room for 3 badges + chat + controls + disconnect
    line_h      = 14

    from factorymind.state import LEADER as _LEADER
    role_map = {r["id"]: r["role"] for r in world_state.get("robots", [])}
    messages = world_state.get("blackboard", [])[-28:][::-1]   # newest first

    for i, msg in enumerate(messages):
        if feed_top + line_h > feed_bottom:
            break

        # Opacity: entries 0-4 full, entries 5-14 fade to 40 %
        if i < 5:
            opacity = 255
        elif i < 15:
            opacity = int(255 - (i - 5) / 10 * (255 - 102))
        else:
            opacity = 102

        msg_type = msg.get("type", "")
        color    = C_MSG.get(msg_type, C_MSG_DEFAULT)
        from_id  = msg.get("from", "?")
        is_user  = (msg_type == "USER")

        if from_id == -1:
            source = "Strategist"
        elif role_map.get(from_id) == _LEADER:
            source = f"Leader-{from_id}"
        else:
            source = f"Worker-{from_id}"

        # mm:ss.cs timestamp
        ts_raw = msg.get("timestamp", 0.0)
        cs     = int((ts_raw % 1) * 100)
        ts_str = time.strftime("%M:%S", time.localtime(ts_raw)) + f".{cs:02d}"

        content = str(msg.get("content", ""))
        rt      = msg.get("route_time")
        if msg_type == "COMPLETE" and rt is not None:
            content += f" [{rt:.2f}s]"
        content = content[:40]

        f_row = pygame.font.SysFont("monospace", 10, bold=is_user)

        ts_s   = f_row.render(ts_str,              True, _dim(C_DIM,   opacity))
        type_s = f_row.render(f"[{msg_type[:4]}]", True, _dim(color,   opacity))
        src_s  = f_row.render(source,              True, _dim(C_DIM,   opacity))
        mc     = (255, 255, 255) if is_user else (200, 200, 215)
        msg_s  = f_row.render(content,             True, _dim(mc,      opacity))

        cx = x0
        screen.blit(ts_s,   (cx, feed_top)); cx += ts_s.get_width()   + 4
        screen.blit(type_s, (cx, feed_top)); cx += type_s.get_width() + 4
        screen.blit(src_s,  (cx, feed_top)); cx += src_s.get_width()  + 4
        screen.blit(msg_s,  (cx, feed_top))
        feed_top += line_h

    # ── Bottom section ─────────────────────────────────────────────────────
    y = feed_bottom
    pygame.draw.line(screen, C_PANEL_EDGE, (x0, y), (PANEL_X + W - PAD, y))
    y += 6

    # Model badges
    for role_str, model_str, col in [
        ("leaders:",    "Nemotron-Nano-9B",  C_ACCENT),
        ("workers:",    "Nemotron-Nano-9B",   C_ACCENT),
        ("strategist:", "Nemotron-Super-49B", (220, 0, 255)),
    ]:
        ls = f_small.render(f"  {role_str}", True, C_DIM)
        ms = f_small.render(f" {model_str}", True, col)
        screen.blit(ls, (x0, y))
        screen.blit(ms, (x0 + ls.get_width(), y))
        y += ls.get_height() + 1

    nv = f_small.render("Powered by NVIDIA NIM  |  ASUS Ascent GX10", True, (65, 65, 95))
    screen.blit(nv, (x0, y))
    y += nv.get_height() + 7

    # Chat input row
    send_w        = 48
    chat_w        = W - PAD * 2 - send_w - 4
    _BTN_CHAT_INPUT = pygame.Rect(x0,                 y, chat_w, 38)
    _BTN_SEND       = pygame.Rect(x0 + chat_w + 4,   y, send_w, 38)

    bdr = C_ACCENT if _chat_focused else (50, 50, 80)
    pygame.draw.rect(screen, (16, 16, 32), _BTN_CHAT_INPUT, border_radius=4)
    pygame.draw.rect(screen, bdr, _BTN_CHAT_INPUT, 2 if _chat_focused else 1, border_radius=4)

    if _chat_text:
        disp = _chat_text[-44:]
        ct   = f_chat.render(disp, True, C_TEXT)
    else:
        ct   = f_chat.render("Give the agents a task...", True, C_DIM)
    screen.blit(ct, (_BTN_CHAT_INPUT.x + 6, _BTN_CHAT_INPUT.y + 11))

    if _chat_focused:
        blink_on = (pygame.time.get_ticks() // 530) % 2 == 0
        if blink_on:
            cursor_x = _BTN_CHAT_INPUT.x + 6 + (ct.get_width() if _chat_text else 0) + 1
            pygame.draw.line(screen, C_ACCENT,
                             (cursor_x, _BTN_CHAT_INPUT.y + 7),
                             (cursor_x, _BTN_CHAT_INPUT.y + 31), 2)

    pygame.draw.rect(screen, (20, 20, 50),  _BTN_SEND, border_radius=4)
    pygame.draw.rect(screen, (60, 60, 100), _BTN_SEND, 1, border_radius=4)
    sl = f_caps.render("SEND", True, C_DIM)
    screen.blit(sl, (_BTN_SEND.x + _BTN_SEND.w // 2 - sl.get_width() // 2,
                     _BTN_SEND.y + _BTN_SEND.h // 2 - sl.get_height() // 2))
    y += 38 + 5

    # RESET | SPEED row
    cbw        = (W - PAD * 2 - 8) // 2
    _BTN_RESET = pygame.Rect(x0,             y, cbw, 24)
    _BTN_SPEED = pygame.Rect(x0 + cbw + 8,   y, cbw, 24)
    for btn, label in [(_BTN_RESET, "RESET"), (_BTN_SPEED, f"SPEED {speed}x →")]:
        pygame.draw.rect(screen, (18, 18, 36), btn, border_radius=4)
        pygame.draw.rect(screen, (50, 50, 80), btn, 1, border_radius=4)
        t = f_caps.render(label, True, C_DIM)
        screen.blit(t, (btn.x + btn.w // 2 - t.get_width() // 2,
                        btn.y + btn.h // 2 - t.get_height() // 2))
    y += 24 + 5

    # DISCONNECT / RECONNECT button
    ticks           = pygame.time.get_ticks()
    pulse           = 0.5 + 0.5 * math.sin(ticks / 350.0)
    _BTN_DISCONNECT = pygame.Rect(x0, y, W - PAD * 2, 36)
    if offline:
        bc  = (40, int(160 + pulse * 80), 60)
        pygame.draw.rect(screen, (10, 35, 18), _BTN_DISCONNECT, border_radius=6)
        pygame.draw.rect(screen, bc,            _BTN_DISCONNECT, 2, border_radius=6)
        lbl = f_btn.render("LOCAL NIM ACTIVE", True, (60, 230, 100))
    else:
        bc  = (int(140 + pulse * 60), 20, 20)
        pygame.draw.rect(screen, (40, 10, 10), _BTN_DISCONNECT, border_radius=6)
        pygame.draw.rect(screen, bc,            _BTN_DISCONNECT, 2, border_radius=6)
        lbl = f_btn.render("FORCE LOCAL NIM", True, (220, 60, 60))
    screen.blit(lbl, (
        _BTN_DISCONNECT.x + _BTN_DISCONNECT.w // 2 - lbl.get_width() // 2,
        _BTN_DISCONNECT.y + _BTN_DISCONNECT.h // 2 - lbl.get_height() // 2,
    ))


# ---------------------------------------------------------------------------
# Disconnect banner
# ---------------------------------------------------------------------------

def draw_disconnect_banner(screen: "pygame.Surface") -> None:
    """Pulsing red banner with GX10 badges below."""
    ticks = pygame.time.get_ticks()
    pulse = 0.5 + 0.5 * math.sin(ticks / 280.0)
    bh    = 36
    r     = int(140 + pulse * 60)

    pygame.draw.rect(screen, (r, 12, 12), (0, 0, screen.get_width(), bh))
    f_b  = pygame.font.SysFont("monospace", 13, bold=True)
    msg  = "LOCAL-ONLY MODE - RUNNING ON ASUS ASCENT GX10"
    surf = f_b.render(msg, True, (255, 220, 220))
    screen.blit(surf, (screen.get_width() // 2 - surf.get_width() // 2, 8))

    f_g  = pygame.font.SysFont("monospace", 11)
    bx   = screen.get_width() // 2 - 220
    by   = bh + 4
    for text in ["✓ Nemotron-9B on GX10", "✓ Nemotron-49B on GX10"]:
        bs    = f_g.render(text, True, (80, 230, 110))
        badge = pygame.Rect(bx, by, bs.get_width() + 14, bs.get_height() + 6)
        pygame.draw.rect(screen, C_BADGE_OK,    badge, border_radius=3)
        pygame.draw.rect(screen, (40, 140, 60), badge, 1, border_radius=3)
        screen.blit(bs, (bx + 7, by + 3))
        bx += badge.w + 12


def _draw_flash_overlay(screen: "pygame.Surface") -> None:
    """Brief red flash on disconnect/reconnect transition."""
    if _disconnect_flash_ticks == 0:
        return
    elapsed = pygame.time.get_ticks() - _disconnect_flash_ticks
    if elapsed >= _FLASH_DURATION_MS:
        return
    alpha   = int(180 * (1.0 - elapsed / _FLASH_DURATION_MS))
    overlay = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
    overlay.fill((*C_FLASH, alpha))
    screen.blit(overlay, (0, 0))


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from factorymind.state import create_initial_state, OPEN_FLOOR

    _ensure_pygame()

    fake_state = create_initial_state(OPEN_FLOOR)
    fake_state["blackboard"] = [
        {"from": 0,  "type": "CLAIM",     "content": "Claiming Parts->Assembly task 3; 4 cells to pickup.", "timestamp": time.time() - 25},
        {"from": -1, "type": "STRATEGY",  "content": "Prioritise QA — route workers south.",                "timestamp": time.time() - 20},
        {"from": 1,  "type": "REASONING", "content": "Evaluating bridge crossing risk at y=25",             "timestamp": time.time() - 15},
        {"from": 2,  "type": "COMPLETE",  "content": "task-1 Parts→QA delivered",                          "timestamp": time.time() - 10, "route_time": 3.42},
        {"from": 3,  "type": "ERROR",     "content": "No path found — bridge blocked",                      "timestamp": time.time() - 6},
        {"from": 0,  "type": "USER",      "content": "Find fastest route between Parts and Shipping",       "timestamp": time.time() - 3},
        {"from": 4,  "type": "FOUND",     "content": "Optimal: Parts->west->Shipping (18 cells, 3.1s)",    "timestamp": time.time() - 1},
    ]
    fake_state["stats"] = {
        "completed": 7, "elapsed": 82.0, "rate": 5.1,
        "last_route_time": 3.42, "avg_route_time": 4.1,
    }
    fake_state["speed_multiplier"] = 2

    screen = init_display()
    clock  = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            result = handle_event(event, fake_state)
            if result == "disconnect":
                s = fake_state["connection_status"]
                fake_state["connection_status"] = "offline" if s == "online" else "online"
            elif result == "mode_builder":
                print("→ BUILDER mode")
            elif result == "mode_cursor":
                print("→ CURSOR mode")
            elif isinstance(result, dict):
                t = result["type"]
                if t == "user_prompt":
                    print(f'User: "{result["text"]}"')
                    fake_state["blackboard"].append({
                        "from": 0, "type": "USER",
                        "content": result["text"],
                        "timestamp": time.time(),
                    })
                elif t == "wall_add":
                    cell = result["cell"]
                    if cell not in fake_state["wall"]:
                        fake_state["wall"].append(cell)
                        fake_state["layout"] = "CUSTOM"
                elif t == "wall_remove":
                    cell = result["cell"]
                    if cell in fake_state["wall"]:
                        fake_state["wall"].remove(cell)

        render(screen, fake_state)
        clock.tick(60)
