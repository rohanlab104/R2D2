"""Pygame renderer for the autonomous FactoryMind factory simulation."""

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

from factorymind import state as S

CELL_SIZE = 16
GRID_OFFSET_X = 50
GRID_OFFSET_Y = 50
PANEL_X = 900
WINDOW_W = 1400
WINDOW_H = 900

C_BG = (10, 10, 20)
C_GRID_FAINT = (14, 14, 23)
C_GRID_MAJOR = (22, 22, 36)
C_ACCENT = (0, 212, 255)
C_TEXT = (224, 224, 224)
C_DIM = (100, 100, 130)
C_PANEL_BG = (12, 12, 24)
C_PANEL_EDGE = (32, 32, 58)
C_WALL = (55, 55, 85)
C_WORKER = (255, 140, 0)
C_LEADER = (0, 212, 255)
C_BELT = (36, 39, 48)
C_FLASH = (200, 0, 0)
C_OK = (0, 180, 90)
C_WARN = (235, 170, 35)
C_DANGER = (220, 60, 60)

C_LOG = {
    "ASSIGN": (0, 212, 255),
    "OBSERVE": (190, 195, 210),
    "RETHINK": (220, 0, 255),
    "COMPLETE": (0, 255, 136),
    "WARNING": (255, 205, 45),
}

_robot_visual_pos: dict[int, list[float]] = {}
_robot_trails: dict[int, deque] = {}
_TRAIL_LEN = 8
_LERP = 0.22

_prev_connection_status = "online"
_disconnect_flash_ticks = 0
_FLASH_DURATION_MS = 500

_mode = "cursor"
_builder_submode = "wall"
_builder_drag_start: Optional[tuple[int, int]] = None
_builder_dragged_cells: set[tuple[int, int]] = set()
# Builder placement rotation (degrees, in-plane). Press R while in
# conveyor/dropbox sub-mode to advance by 45 deg. Wraps at 360.
_builder_rotation: float = 0.0

_BTN_DISCONNECT: Optional["pygame.Rect"] = None
_BTN_MODE_CURSOR: Optional["pygame.Rect"] = None
_BTN_MODE_BUILDER: Optional["pygame.Rect"] = None
_BTN_SPEED: Optional["pygame.Rect"] = None
_BTN_START: Optional["pygame.Rect"] = None
_BTN_STOP: Optional["pygame.Rect"] = None
_BTN_RESET: Optional["pygame.Rect"] = None
_BTN_BUILDER: dict[str, "pygame.Rect"] = {}


def _ensure_pygame() -> None:
    if pygame is None:
        raise ImportError("pygame is required. Run: pip3 install pygame")


def init_display(width: int = WINDOW_W, height: int = WINDOW_H) -> "pygame.Surface":
    _ensure_pygame()
    pygame.init()
    pygame.display.set_caption("FactoryMind 3D - Autonomous Factory")
    return pygame.display.set_mode((width, height))


def _grid_cell_at(pos: tuple[int, int]) -> Optional[tuple[int, int]]:
    px, py = pos
    gx = (px - GRID_OFFSET_X) // CELL_SIZE
    gy = (py - GRID_OFFSET_Y) // CELL_SIZE
    if 0 <= gx < S.GRID_WIDTH and 0 <= gy < S.GRID_HEIGHT:
        return (gx, gy)
    return None


def _line_cells(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
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
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return cells


def _dim(color: tuple[int, int, int], opacity: int) -> tuple[int, int, int]:
    return tuple(int(c * opacity // 255) for c in color)  # type: ignore[return-value]


def _cell_center(cell: list[int] | tuple[int, int]) -> tuple[int, int]:
    x, y = cell
    return (
        GRID_OFFSET_X + x * CELL_SIZE + CELL_SIZE // 2,
        GRID_OFFSET_Y + y * CELL_SIZE + CELL_SIZE // 2,
    )


def _cell_rect(cell: list[int] | tuple[int, int], w: int = 1, h: int = 1) -> "pygame.Rect":
    x, y = cell
    return pygame.Rect(
        GRID_OFFSET_X + x * CELL_SIZE,
        GRID_OFFSET_Y + y * CELL_SIZE,
        w * CELL_SIZE,
        h * CELL_SIZE,
    )


def _draw_alpha_circle(surface: "pygame.Surface", color: tuple[int, int, int],
                       center: tuple[int, int], radius: int, alpha: int) -> None:
    size = radius * 2 + 2
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.circle(s, (*color, alpha), (radius + 1, radius + 1), radius)
    surface.blit(s, (center[0] - radius - 1, center[1] - radius - 1))


def render(screen: "pygame.Surface", world_state: dict) -> None:
    """Draw one complete frame from world_state and flip the display."""
    global _prev_connection_status, _disconnect_flash_ticks
    _ensure_pygame()

    current = world_state.get("connection_status", "online")
    if current != _prev_connection_status:
        _disconnect_flash_ticks = pygame.time.get_ticks()
        _prev_connection_status = current

    pygame.mouse.set_visible(True)
    screen.fill(C_BG)
    wall_set = {tuple(c) for c in world_state.get("wall", [])}

    draw_grid(screen)
    draw_walls(screen, wall_set)
    draw_factories(screen, world_state)
    draw_dropboxes(screen, world_state)
    draw_leader_tower(screen, world_state)
    draw_workers(screen, world_state)
    draw_builder_overlay(screen, world_state)
    draw_sidepanel(screen, world_state)

    if current == "offline":
        draw_disconnect_banner(screen)
    _draw_flash_overlay(screen)
    pygame.display.flip()


def handle_event(event: "pygame.event.Event", world_state: dict) -> "Union[dict, str, None]":
    """Process one pygame event and return an action for main.py."""
    global _builder_drag_start, _builder_dragged_cells, _builder_rotation

    if event.type == pygame.MOUSEBUTTONDOWN:
        pos = event.pos
        button = event.button
        if button == 1:
            action = _check_panel_buttons(pos, world_state)
            if action is not None:
                return action
            if _mode == "builder":
                cell = _grid_cell_at(pos)
                if not cell:
                    return None
                if _builder_submode == "wall":
                    _builder_drag_start = cell
                    _builder_dragged_cells = {cell}
                    return {"type": "wall_add", "cell": list(cell)}
                if _builder_submode == "factory":
                    return {
                        "type": "place_factory",
                        "cell": list(cell),
                        "color": _next_factory_color(world_state),
                        "rotation_y": math.radians(_builder_rotation),
                    }
                if _builder_submode == "dropbox":
                    return {
                        "type": "place_dropbox",
                        "cell": list(cell),
                        "color": _next_dropbox_color(world_state),
                        "rotation_y": math.radians(_builder_rotation),
                    }
                if _builder_submode == "worker":
                    return {"type": "place_worker", "cell": list(cell)}
        elif button == 3 and _mode == "builder":
            cell = _grid_cell_at(pos)
            if cell:
                return {"type": "wall_remove", "cell": list(cell)}

    elif event.type == pygame.MOUSEBUTTONUP:
        if event.button == 1:
            _builder_drag_start = None
            _builder_dragged_cells = set()

    elif event.type == pygame.MOUSEMOTION:
        if _mode == "builder" and _builder_submode == "wall":
            if _builder_drag_start and pygame.mouse.get_pressed()[0]:
                cell = _grid_cell_at(event.pos)
                if cell and cell not in _builder_dragged_cells:
                    _builder_dragged_cells.add(cell)
                    return {"type": "wall_add", "cell": list(cell)}

    elif event.type == pygame.KEYDOWN:
        # R while in conveyor/dropbox sub-mode rotates the placement preview
        # 45 deg around the y axis (i.e., in the top-down XY plane).
        if event.key == pygame.K_r and _mode == "builder" and _builder_submode in ("factory", "dropbox"):
            _builder_rotation = (_builder_rotation + 45.0) % 360.0

    return None


def get_button_click(pos: tuple[int, int]) -> Optional[str]:
    return _check_panel_buttons(pos, {})


def _check_panel_buttons(pos: tuple[int, int], world_state: dict) -> Optional[str]:
    global _mode, _builder_submode
    if _BTN_DISCONNECT and _BTN_DISCONNECT.collidepoint(pos):
        return "reconnect" if world_state.get("connection_status") == "offline" else "disconnect"
    if _BTN_START and _BTN_START.collidepoint(pos):
        return "start"
    if _BTN_STOP and _BTN_STOP.collidepoint(pos):
        return "stop"
    if _BTN_RESET and _BTN_RESET.collidepoint(pos):
        return "reset"
    if _BTN_SPEED and _BTN_SPEED.collidepoint(pos):
        return "speed_toggle"
    if _BTN_MODE_CURSOR and _BTN_MODE_CURSOR.collidepoint(pos):
        _mode = "cursor"
        return "mode_cursor"
    if _BTN_MODE_BUILDER and _BTN_MODE_BUILDER.collidepoint(pos):
        _mode = "builder"
        return "mode_builder"
    for key, rect in _BTN_BUILDER.items():
        if rect.collidepoint(pos):
            _builder_submode = key
            return f"builder_{key}"
    return None


def _next_factory_color(world_state: dict) -> str:
    return S.next_color_name(len(world_state.get("factories", [])))


def _next_dropbox_color(world_state: dict) -> str:
    factory_colors = [f.get("color") for f in world_state.get("factories", [])]
    box_colors = [b.get("color") for b in world_state.get("dropboxes", [])]
    for color in factory_colors:
        if color not in box_colors:
            return str(color)
    if factory_colors:
        return str(factory_colors[len(box_colors) % len(factory_colors)])
    return S.next_color_name(len(box_colors))


def draw_grid(screen: "pygame.Surface") -> None:
    for x in range(S.GRID_WIDTH + 1):
        color = C_GRID_MAJOR if x % 5 == 0 else C_GRID_FAINT
        px = GRID_OFFSET_X + x * CELL_SIZE
        pygame.draw.line(screen, color, (px, GRID_OFFSET_Y),
                         (px, GRID_OFFSET_Y + S.GRID_HEIGHT * CELL_SIZE))
    for y in range(S.GRID_HEIGHT + 1):
        color = C_GRID_MAJOR if y % 5 == 0 else C_GRID_FAINT
        py = GRID_OFFSET_Y + y * CELL_SIZE
        pygame.draw.line(screen, color, (GRID_OFFSET_X, py),
                         (GRID_OFFSET_X + S.GRID_WIDTH * CELL_SIZE, py))


def draw_walls(screen: "pygame.Surface", wall_set: set[tuple[int, int]]) -> None:
    for cell in wall_set:
        rect = _cell_rect(cell)
        pygame.draw.rect(screen, C_WALL, rect)
        pygame.draw.rect(screen, (80, 80, 110), rect, 1)


def draw_factories(screen: "pygame.Surface", world_state: dict) -> None:
    f_label = pygame.font.SysFont("monospace", 10, bold=True)
    f_small = pygame.font.SysFont("monospace", 9, bold=True)
    ticks = pygame.time.get_ticks()
    for factory in world_state.get("factories", []):
        color = tuple(factory.get("color_rgb", [180, 180, 180]))
        pos = factory["pos"]
        rect = _cell_rect(pos, 3, 3)
        fill = tuple(max(0, c - 125) for c in color)
        border = tuple(min(255, c + 25) for c in color)
        pygame.draw.rect(screen, fill, rect, border_radius=4)
        pygame.draw.rect(screen, border, rect, 2, border_radius=4)
        pygame.draw.rect(screen, (18, 18, 28), rect.inflate(-12, -12), border_radius=3)
        label = f_label.render(factory["name"], True, C_TEXT)
        screen.blit(label, (rect.centerx - label.get_width() // 2, rect.y - 13))

        belt_start = factory["belt_start"]
        belt_len = int(factory.get("belt_length", 5))
        belt_rect = _cell_rect(belt_start, belt_len, 1)
        pygame.draw.rect(screen, C_BELT, belt_rect, border_radius=3)
        pygame.draw.rect(screen, (80, 85, 98), belt_rect, 1, border_radius=3)
        offset = (ticks // 90) % 12
        for i in range(-1, belt_len * CELL_SIZE // 12 + 2):
            x = belt_rect.x + i * 12 + offset
            pygame.draw.line(screen, (120, 125, 140),
                             (x, belt_rect.y + 3), (x + 7, belt_rect.y + belt_rect.h - 3), 2)

        for pkg in factory.get("conveyor", []):
            progress = float(pkg.get("progress", 0.0))
            cx = belt_rect.x + int(progress * CELL_SIZE)
            cy = belt_rect.centery
            _draw_package_cube(screen, color, (cx, cy), 8)

        pad = factory["drop_pad"]
        pad_rect = _cell_rect(pad)
        pygame.draw.rect(screen, (28, 30, 36), pad_rect, border_radius=3)
        pygame.draw.rect(screen, border, pad_rect, 2, border_radius=3)
        packages = factory.get("pad_packages", [])
        for index, pkg in enumerate(packages[:5]):
            px = pad_rect.centerx - 4 + (index % 2) * 5
            py = pad_rect.centery - 5 - index * 2
            _draw_package_cube(screen, tuple(pkg.get("color_rgb", color)), (px, py), 7)
        if len(packages) > 5:
            more = f_small.render(f"+{len(packages) - 5}", True, C_TEXT)
            screen.blit(more, (pad_rect.right + 2, pad_rect.y - 1))


def draw_dropboxes(screen: "pygame.Surface", world_state: dict) -> None:
    font = pygame.font.SysFont("monospace", 9, bold=True)
    for box in world_state.get("dropboxes", []):
        color = tuple(box.get("color_rgb", [180, 180, 180]))
        x, y = box["pos"]
        rect = _cell_rect([x, y], 3, 2)
        points = [
            (rect.x + 3, rect.y + 5),
            (rect.right - 3, rect.y + 5),
            (rect.right - 7, rect.bottom - 2),
            (rect.x + 7, rect.bottom - 2),
        ]
        pygame.draw.polygon(screen, (18, 18, 28), points)
        pygame.draw.lines(screen, color, True, points, 2)
        pygame.draw.line(screen, _dim(color, 140), (rect.x + 7, rect.y + 11),
                         (rect.right - 7, rect.y + 11), 2)
        label = font.render(box["name"], True, C_TEXT)
        screen.blit(label, (rect.centerx - label.get_width() // 2, rect.y - 12))
        count = font.render(f"{box.get('delivered', 0)} delivered", True, color)
        screen.blit(count, (rect.centerx - count.get_width() // 2, rect.bottom + 1))


def draw_leader_tower(screen: "pygame.Surface", world_state: dict) -> None:
    leader = world_state.get("leader", {})
    pos = leader.get("pos", [25, 47])
    cx, cy = _cell_center(pos)
    thinking = bool(world_state.get("leader_thinking") or leader.get("thinking"))
    pulse = 0.5 + 0.5 * math.sin(pygame.time.get_ticks() / 180.0)
    if thinking:
        _draw_alpha_circle(screen, C_ACCENT, (cx, cy - 12), int(22 + pulse * 10), 70)
    base = pygame.Rect(cx - 13, cy - 24, 26, 40)
    pygame.draw.rect(screen, (8, 40, 55), base, border_radius=6)
    pygame.draw.rect(screen, C_ACCENT, base, 2, border_radius=6)
    pygame.draw.circle(screen, C_ACCENT, (cx, cy - 28), 7)
    pygame.draw.line(screen, C_ACCENT, (cx, cy - 35), (cx, cy - 48), 2)
    pygame.draw.circle(screen, (210, 250, 255), (cx, cy - 50), 3)
    font = pygame.font.SysFont("monospace", 10, bold=True)
    label = font.render("LEADER", True, C_ACCENT)
    screen.blit(label, (cx - label.get_width() // 2, cy - 65))


def draw_workers(screen: "pygame.Surface", world_state: dict) -> None:
    font = pygame.font.SysFont("monospace", 9, bold=True)
    ticks = pygame.time.get_ticks()
    for worker in world_state.get("workers", []):
        rid = worker["id"]
        gx, gy = worker["pos"]
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

        trail = list(_robot_trails[rid])
        for i, (tx, ty) in enumerate(trail[:-1]):
            frac = (i + 1) / max(1, len(trail))
            _draw_alpha_circle(screen, C_WORKER, (tx, ty), max(1, int(1 + frac * 4)),
                               int(15 + frac * 110))
        radius = CELL_SIZE // 2 - 1
        if worker.get("status") == "idle":
            pulse = 0.5 + 0.5 * math.sin(ticks / 500.0 + rid)
            _draw_alpha_circle(screen, C_WORKER, (px, py), int(radius + 4 + pulse * 3), 30)
        pygame.draw.circle(screen, C_WORKER, (px, py), radius)
        pygame.draw.circle(screen, (0, 0, 0), (px, py), radius, 1)
        lbl = font.render(str(rid), True, (0, 0, 0))
        screen.blit(lbl, (px - lbl.get_width() // 2, py - lbl.get_height() // 2))
        carrying = worker.get("carrying")
        if carrying:
            _draw_package_cube(screen, tuple(carrying.get("color_rgb", [180, 180, 180])), (px + 4, py - 11), 8)


def _draw_package_cube(screen: "pygame.Surface", color: tuple[int, int, int],
                       center: tuple[int, int], size: int) -> None:
    x, y = center
    rect = pygame.Rect(x - size // 2, y - size // 2, size, size)
    pygame.draw.rect(screen, color, rect, border_radius=1)
    pygame.draw.line(screen, tuple(min(255, c + 55) for c in color),
                     (rect.x, rect.y), (rect.right - 1, rect.y), 1)
    pygame.draw.rect(screen, (0, 0, 0), rect, 1)


def _rotate_points(points: list[tuple[float, float]], pivot: tuple[float, float],
                   degrees: float) -> list[tuple[int, int]]:
    """Rotate a polygon's points around ``pivot`` by ``degrees``."""
    angle = math.radians(degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    px, py = pivot
    out: list[tuple[int, int]] = []
    for x, y in points:
        dx = x - px
        dy = y - py
        rx = dx * cos_a - dy * sin_a + px
        ry = dx * sin_a + dy * cos_a + py
        out.append((int(round(rx)), int(round(ry))))
    return out


def draw_builder_overlay(screen: "pygame.Surface", world_state: dict) -> None:
    if _mode != "builder":
        return
    mouse_pos = pygame.mouse.get_pos()
    hover = _grid_cell_at(mouse_pos)
    wall_set = {tuple(c) for c in world_state.get("wall", [])}
    if _builder_submode == "wall" and _builder_drag_start and hover and hover != _builder_drag_start:
        for cell in _line_cells(*_builder_drag_start, *hover):
            if cell not in _builder_dragged_cells and cell not in wall_set:
                overlay = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                overlay.fill((0, 212, 255, 45))
                screen.blit(overlay, _cell_rect(cell).topleft)
    if not hover:
        return
    rect = _cell_rect(hover)
    if _builder_submode == "wall":
        pygame.draw.rect(screen, (*C_ACCENT, 90), rect)
        pygame.draw.rect(screen, C_ACCENT, rect, 2)
    elif _builder_submode == "factory":
        _draw_factory_outline(screen, hover, _builder_rotation)
        _draw_rotation_chip(screen, hover, _builder_rotation)
    elif _builder_submode == "dropbox":
        _draw_dropbox_outline(screen, hover, _builder_rotation)
        _draw_rotation_chip(screen, hover, _builder_rotation)
    elif _builder_submode == "worker":
        pygame.draw.circle(screen, C_WORKER, _cell_center(hover), CELL_SIZE // 2, 2)


def _draw_factory_outline(screen: "pygame.Surface", cell: tuple[int, int],
                          rotation_deg: float) -> None:
    """Cyan schematic of a factory + its right-facing conveyor + drop pad."""
    x, y = cell
    base_px = GRID_OFFSET_X + x * CELL_SIZE
    base_py = GRID_OFFSET_Y + y * CELL_SIZE
    # Pivot at the factory body center (it's a 3x3 starting at the hover cell).
    pivot = (base_px + 1.5 * CELL_SIZE, base_py + 1.5 * CELL_SIZE)

    # Factory body — 3x3 box.
    body = [
        (base_px,                  base_py),
        (base_px + 3 * CELL_SIZE,  base_py),
        (base_px + 3 * CELL_SIZE,  base_py + 3 * CELL_SIZE),
        (base_px,                  base_py + 3 * CELL_SIZE),
    ]
    # Conveyor belt — 5 cells extending right of the body, vertically centered.
    belt_y0 = base_py + CELL_SIZE
    belt = [
        (base_px + 3 * CELL_SIZE,        belt_y0),
        (base_px + 8 * CELL_SIZE,        belt_y0),
        (base_px + 8 * CELL_SIZE,        belt_y0 + CELL_SIZE),
        (base_px + 3 * CELL_SIZE,        belt_y0 + CELL_SIZE),
    ]
    # Drop pad — single cell at end of belt.
    pad = [
        (base_px + 8 * CELL_SIZE,        belt_y0),
        (base_px + 9 * CELL_SIZE,        belt_y0),
        (base_px + 9 * CELL_SIZE,        belt_y0 + CELL_SIZE),
        (base_px + 8 * CELL_SIZE,        belt_y0 + CELL_SIZE),
    ]
    accent = (*C_ACCENT, 220)
    accent_dim = (*C_ACCENT, 110)

    # Translucent body fill for legibility.
    body_pts = _rotate_points([(p[0], p[1]) for p in body], pivot, rotation_deg)
    belt_pts = _rotate_points([(p[0], p[1]) for p in belt], pivot, rotation_deg)
    pad_pts = _rotate_points([(p[0], p[1]) for p in pad], pivot, rotation_deg)
    surf = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    pygame.draw.polygon(surf, (*C_ACCENT, 45), body_pts)
    pygame.draw.polygon(surf, (*C_ACCENT, 35), belt_pts)
    pygame.draw.polygon(surf, (*C_ACCENT, 60), pad_pts)
    pygame.draw.polygon(surf, accent, body_pts, 2)
    pygame.draw.polygon(surf, accent_dim, belt_pts, 1)
    pygame.draw.polygon(surf, accent, pad_pts, 2)
    # Arrow along the conveyor direction (rotates with the belt).
    arrow_start = (base_px + 3.4 * CELL_SIZE, belt_y0 + CELL_SIZE / 2)
    arrow_end   = (base_px + 7.6 * CELL_SIZE, belt_y0 + CELL_SIZE / 2)
    head_a      = (base_px + 7.0 * CELL_SIZE, belt_y0 + CELL_SIZE / 2 - 4)
    head_b      = (base_px + 7.0 * CELL_SIZE, belt_y0 + CELL_SIZE / 2 + 4)
    rot_pts = _rotate_points([arrow_start, arrow_end, head_a, head_b], pivot, rotation_deg)
    pygame.draw.line(surf, accent, rot_pts[0], rot_pts[1], 2)
    pygame.draw.line(surf, accent, rot_pts[1], rot_pts[2], 2)
    pygame.draw.line(surf, accent, rot_pts[1], rot_pts[3], 2)
    screen.blit(surf, (0, 0))


def _draw_dropbox_outline(screen: "pygame.Surface", cell: tuple[int, int],
                          rotation_deg: float) -> None:
    """Cyan schematic of a drop box (3 wide x 2 deep) with rotation."""
    x, y = cell
    base_px = GRID_OFFSET_X + x * CELL_SIZE
    base_py = GRID_OFFSET_Y + y * CELL_SIZE
    pivot = (base_px + 1.5 * CELL_SIZE, base_py + 1.0 * CELL_SIZE)
    # Footprint: 3x2 cells.
    footprint = [
        (base_px,                 base_py),
        (base_px + 3 * CELL_SIZE, base_py),
        (base_px + 3 * CELL_SIZE, base_py + 2 * CELL_SIZE),
        (base_px,                 base_py + 2 * CELL_SIZE),
    ]
    # Trapezoidal lid mimicking the live render.
    lid = [
        (base_px + 4,                  base_py + 5),
        (base_px + 3 * CELL_SIZE - 4,  base_py + 5),
        (base_px + 3 * CELL_SIZE - 8,  base_py + 2 * CELL_SIZE - 2),
        (base_px + 8,                  base_py + 2 * CELL_SIZE - 2),
    ]
    surf = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    fp_pts  = _rotate_points([(p[0], p[1]) for p in footprint], pivot, rotation_deg)
    lid_pts = _rotate_points([(p[0], p[1]) for p in lid], pivot, rotation_deg)
    pygame.draw.polygon(surf, (*C_ACCENT, 35), fp_pts)
    pygame.draw.polygon(surf, (*C_ACCENT, 55), lid_pts)
    pygame.draw.polygon(surf, (*C_ACCENT, 220), fp_pts, 2)
    pygame.draw.polygon(surf, (*C_ACCENT, 220), lid_pts, 2)
    # Open-box "front lip" line.
    lip_start = (base_px + 8,                base_py + CELL_SIZE + 3)
    lip_end   = (base_px + 3 * CELL_SIZE - 8, base_py + CELL_SIZE + 3)
    rot = _rotate_points([lip_start, lip_end], pivot, rotation_deg)
    pygame.draw.line(surf, (*C_ACCENT, 220), rot[0], rot[1], 2)
    screen.blit(surf, (0, 0))


def _draw_rotation_chip(screen: "pygame.Surface", cell: tuple[int, int],
                        rotation_deg: float) -> None:
    """Small chip showing current rotation + the R hint."""
    font = pygame.font.SysFont("monospace", 9, bold=True)
    label = f"R={int(rotation_deg):3d} deg  (press R)"
    text = font.render(label, True, C_ACCENT)
    px = GRID_OFFSET_X + cell[0] * CELL_SIZE
    py = GRID_OFFSET_Y + cell[1] * CELL_SIZE - 14
    bg = pygame.Rect(px - 2, py - 1, text.get_width() + 8, text.get_height() + 2)
    pygame.draw.rect(screen, (10, 16, 28), bg, border_radius=3)
    pygame.draw.rect(screen, C_ACCENT, bg, 1, border_radius=3)
    screen.blit(text, (px + 2, py))


def draw_sidepanel(screen: "pygame.Surface", world_state: dict) -> None:
    global _BTN_DISCONNECT, _BTN_MODE_CURSOR, _BTN_MODE_BUILDER, _BTN_SPEED
    global _BTN_START, _BTN_STOP, _BTN_RESET, _BTN_BUILDER

    W = screen.get_width() - PANEL_X
    H = screen.get_height()
    PAD = 14
    x0 = PANEL_X + PAD
    pygame.draw.rect(screen, C_PANEL_BG, (PANEL_X, 0, W, H))
    pygame.draw.line(screen, C_PANEL_EDGE, (PANEL_X, 0), (PANEL_X, H))

    f_title = pygame.font.SysFont("monospace", 20, bold=True)
    f_caps = pygame.font.SysFont("monospace", 10, bold=True)
    f_body = pygame.font.SysFont("monospace", 12)
    f_big = pygame.font.SysFont("monospace", 19, bold=True)
    f_small = pygame.font.SysFont("monospace", 10)
    f_btn = pygame.font.SysFont("monospace", 11, bold=True)

    y = 14
    title = f_title.render("FACTORYMIND 3D", True, C_ACCENT)
    screen.blit(title, (x0, y))
    y += title.get_height() + 2
    sub = f_small.render("Nemotron + DGX Spark GX10", True, C_DIM)
    screen.blit(sub, (x0, y))
    y += sub.get_height() + 5
    pygame.draw.line(screen, C_ACCENT, (x0, y), (PANEL_X + W - PAD, y), 1)
    y += 8

    stats = world_state.get("stats", {})
    elapsed = stats.get("elapsed", 0.0)
    last_rt = stats.get("last_route_time")
    tiles = [
        ("COMPLETED", str(stats.get("completed", 0)), ""),
        ("TIME", f"{int(elapsed)//60:02d}:{int(elapsed)%60:02d}", ""),
        ("RATE/MIN", f"{stats.get('rate', 0.0):.1f}", ""),
        ("LAST ROUTE", f"{last_rt:.2f}s" if last_rt else "--", ""),
    ]
    tw = (W - PAD * 2 - 12) // 4
    for i, (label, value, subtext) in enumerate(tiles):
        rect = pygame.Rect(x0 + i * (tw + 4), y, tw, 52)
        pygame.draw.rect(screen, (18, 18, 36), rect, border_radius=4)
        pygame.draw.rect(screen, (44, 44, 76), rect, 1, border_radius=4)
        ls = f_small.render(label, True, C_DIM)
        vs = f_big.render(value, True, C_ACCENT)
        screen.blit(ls, (rect.centerx - ls.get_width() // 2, rect.y + 4))
        screen.blit(vs, (rect.centerx - vs.get_width() // 2, rect.y + 18))
        if subtext:
            ss = f_small.render(subtext, True, C_DIM)
            screen.blit(ss, (rect.centerx - ss.get_width() // 2, rect.y + 39))
    y += 59

    mbw = (W - PAD * 2 - 8) // 2
    _BTN_MODE_CURSOR = pygame.Rect(x0, y, mbw, 26)
    _BTN_MODE_BUILDER = pygame.Rect(x0 + mbw + 8, y, mbw, 26)
    for rect, label, key in [(_BTN_MODE_CURSOR, "CURSOR", "cursor"), (_BTN_MODE_BUILDER, "BUILDER", "builder")]:
        active = _mode == key
        _draw_button(screen, rect, label, C_ACCENT if active else C_DIM, active=active)
    y += 33

    _BTN_BUILDER = {}
    if _mode == "builder":
        bw = (W - PAD * 2 - 12) // 4
        for i, key in enumerate(["wall", "factory", "dropbox", "worker"]):
            rect = pygame.Rect(x0 + i * (bw + 4), y, bw, 26)
            _BTN_BUILDER[key] = rect
            _draw_button(screen, rect, key.upper(), C_ACCENT if _builder_submode == key else C_DIM,
                         active=_builder_submode == key)
        y += 33

    _BTN_SPEED = pygame.Rect(x0, y, W - PAD * 2, 25)
    _draw_button(screen, _BTN_SPEED, f"SPEED {world_state.get('speed_multiplier', 1)}x", C_DIM)
    y += 31

    target_lines = str(world_state.get("inference_target", "")).splitlines()[:3]
    pygame.draw.rect(screen, (14, 14, 28), (x0, y, W - PAD * 2, 45), border_radius=4)
    pygame.draw.rect(screen, C_PANEL_EDGE, (x0, y, W - PAD * 2, 45), 1, border_radius=4)
    inf = target_lines or ["Inference target: loading"]
    for line in inf:
        txt = f_small.render(line[:66], True, C_DIM)
        screen.blit(txt, (x0 + 6, y + 5))
        y += 13
    y += 7
    pygame.draw.line(screen, C_PANEL_EDGE, (x0, y), (PANEL_X + W - PAD, y))
    y += 8

    header = f_body.render("LEADER THOUGHT PROCESS", True, C_ACCENT)
    screen.blit(header, (x0, y))
    state_label = "RUNNING" if world_state.get("running") else "STOPPED"
    st = f_small.render(state_label, True, C_OK if world_state.get("running") else C_WARN)
    screen.blit(st, (PANEL_X + W - PAD - st.get_width(), y + 2))
    y += header.get_height() + 6

    feed_bottom = H - 190
    draw_thought_log(screen, world_state, x0, y, PANEL_X + W - PAD, feed_bottom)
    y = feed_bottom + 7
    pygame.draw.line(screen, C_PANEL_EDGE, (x0, y), (PANEL_X + W - PAD, y))
    y += 7

    worker_header = f_body.render("WORKER STATUS", True, C_ACCENT)
    screen.blit(worker_header, (x0, y))
    y += worker_header.get_height() + 3
    for worker in world_state.get("workers", [])[:5]:
        status = _worker_status_text(worker, world_state)
        txt = f_small.render(status[:68], True, C_TEXT if worker.get("status") != "idle" else C_DIM)
        screen.blit(txt, (x0, y))
        y += 13
    y += 3
    pygame.draw.line(screen, C_PANEL_EDGE, (x0, y), (PANEL_X + W - PAD, y))
    y += 8

    bw = (W - PAD * 2 - 12) // 3
    _BTN_START = pygame.Rect(x0, y, bw, 32)
    _BTN_STOP = pygame.Rect(x0 + bw + 6, y, bw, 32)
    _BTN_RESET = pygame.Rect(x0 + 2 * (bw + 6), y, bw, 32)
    _draw_button(screen, _BTN_START, "START", (60, 230, 100), fill=(8, 42, 20))
    _draw_button(screen, _BTN_STOP, "STOP", C_WARN, fill=(48, 32, 8))
    _draw_button(screen, _BTN_RESET, "RESET", C_DANGER, fill=(32, 10, 10))

    offline = world_state.get("connection_status") == "offline"
    _BTN_DISCONNECT = pygame.Rect(x0, H - 50, W - PAD * 2, 36)
    label = "RECONNECT CLOUD" if offline else "DISCONNECT CLOUD"
    color = (60, 230, 100) if offline else (220, 60, 60)
    fill = (10, 35, 18) if offline else (40, 10, 10)
    _draw_button(screen, _BTN_DISCONNECT, label, color, fill=fill, pulse=True)


def _draw_button(screen: "pygame.Surface", rect: "pygame.Rect", label: str,
                 color: tuple[int, int, int], *, fill: tuple[int, int, int] = (18, 18, 36),
                 active: bool = False, pulse: bool = False) -> None:
    border = color if active else (52, 52, 84)
    if pulse:
        p = 0.5 + 0.5 * math.sin(pygame.time.get_ticks() / 360.0)
        border = tuple(min(255, int(c * (0.65 + p * 0.45))) for c in color)
    pygame.draw.rect(screen, fill, rect, border_radius=4)
    pygame.draw.rect(screen, border, rect, 2 if active else 1, border_radius=4)
    font = pygame.font.SysFont("monospace", 10, bold=True)
    text = font.render(label, True, color)
    screen.blit(text, (rect.centerx - text.get_width() // 2, rect.centery - text.get_height() // 2))


def draw_thought_log(screen: "pygame.Surface", world_state: dict,
                     x0: int, top: int, right: int, bottom: int) -> None:
    font = pygame.font.SysFont("monospace", 10)
    y = top
    messages = world_state.get("thought_log", [])[-34:][::-1]
    for i, msg in enumerate(messages):
        if y + 14 > bottom:
            break
        opacity = 255 if i < 5 else max(95, int(255 - (i - 4) * 18))
        msg_type = str(msg.get("type", "OBSERVE"))
        color = C_LOG.get(msg_type, C_DIM)
        elapsed = float(msg.get("elapsed", 0.0))
        ts = f"{int(elapsed)//60:02d}:{int(elapsed)%60:02d}.{int((elapsed % 1) * 100):02d}"
        agent = str(msg.get("agent", "Leader"))
        content = str(msg.get("content", ""))[:58]
        row = f"{ts} [{msg_type[:7]}] {agent}: {content}"
        surf = font.render(row, True, _dim(color, opacity))
        screen.blit(surf, (x0, y))
        y += 14


def _worker_status_text(worker: dict, world_state: dict) -> str:
    carrying = worker.get("carrying")
    carry = f" [carrying {carrying['color']}]" if carrying else ""
    status = worker.get("status", "idle")
    if status == "to_pickup":
        factory = _factory_name(world_state, worker.get("target_factory_id"))
        return f"{worker['name']}: en route to {factory}{carry}"
    if status == "to_dropbox":
        box = _dropbox_name(world_state, worker.get("target_dropbox_id"))
        return f"{worker['name']}: en route to {box}{carry}"
    if status == "picking":
        return f"{worker['name']}: picking up package{carry}"
    if status == "dropping":
        return f"{worker['name']}: depositing package{carry}"
    return f"{worker['name']}: idle, awaiting assignment"


def _factory_name(world_state: dict, factory_id: int | None) -> str:
    factory = next((f for f in world_state.get("factories", []) if f.get("id") == factory_id), None)
    return factory.get("name", "factory") if factory else "factory"


def _dropbox_name(world_state: dict, box_id: int | None) -> str:
    box = next((b for b in world_state.get("dropboxes", []) if b.get("id") == box_id), None)
    return box.get("name", "drop box") if box else "drop box"


def draw_disconnect_banner(screen: "pygame.Surface") -> None:
    ticks = pygame.time.get_ticks()
    pulse = 0.5 + 0.5 * math.sin(ticks / 280.0)
    r = int(140 + pulse * 60)
    pygame.draw.rect(screen, (r, 12, 12), (0, 0, screen.get_width(), 36))
    font = pygame.font.SysFont("monospace", 13, bold=True)
    msg = "CLOUD DISCONNECTED - LOCAL GX10 NIM POLICY ACTIVE"
    surf = font.render(msg, True, (255, 220, 220))
    screen.blit(surf, (screen.get_width() // 2 - surf.get_width() // 2, 8))


def _draw_flash_overlay(screen: "pygame.Surface") -> None:
    if _disconnect_flash_ticks == 0:
        return
    elapsed = pygame.time.get_ticks() - _disconnect_flash_ticks
    if elapsed >= _FLASH_DURATION_MS:
        return
    alpha = int(180 * (1.0 - elapsed / _FLASH_DURATION_MS))
    overlay = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
    overlay.fill((*C_FLASH, alpha))
    screen.blit(overlay, (0, 0))


if __name__ == "__main__":
    fake_state = S.create_initial_state()
    fake_state["running"] = True
    fake_state["inference_target"] = "leader -> localhost:8001\nworker -> localhost:8000"
    fake_state["thought_log"] = [
        {"elapsed": 0.1, "type": "OBSERVE", "agent": "Leader", "content": "2 packages on Factory-A pad"},
        {"elapsed": 0.8, "type": "ASSIGN", "agent": "Leader", "content": "Worker-1 -> Factory-A then DropBox-Red"},
    ]
    screen = init_display()
    clock = pygame.time.Clock()
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            handle_event(ev, fake_state)
        render(screen, fake_state)
        clock.tick(60)
