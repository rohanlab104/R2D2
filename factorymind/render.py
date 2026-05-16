"""Pygame renderer for FactoryMind R2D2 — visual upgrade v2.

Person B owns this file.

Design intent: ops-dashboard aesthetic. When a judge walks up, the first
impression must read "live autonomous system", not "homework project."
All six visual upgrades are implemented here. Only reads from world_state —
no writes, no side effects outside module-level animation state.
"""

from __future__ import annotations

import math
import sys
import time
from collections import deque
from typing import Optional

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
# Colour palette — Upgrade 1
# ---------------------------------------------------------------------------
C_BG          = (10,  10,  20)   # #0a0a14 dark blue-black
C_GRID_FAINT  = (26,  26,  42)   # #1a1a2a subtle grid
C_GRID_MAJOR  = (37,  37,  64)   # #252540 every 5th line
C_ACCENT      = (0,   212, 255)  # #00d4ff cyan
C_TEXT        = (224, 224, 224)
C_DIM         = (100, 100, 130)
C_LEADER      = (0,   212, 255)  # cyan robots
C_WORKER      = (255, 140, 0)    # orange robots
C_WALL        = (55,  55,  85)
C_PANEL_BG    = (12,  12,  24)
C_PANEL_EDGE  = (28,  28,  55)
C_WHITE       = (255, 255, 255)
C_FLASH       = (200, 0,   0)
C_BANNER_BASE = (160, 15,  15)
C_BADGE_OK    = (10,  80,  30)

# Message type colours — Upgrade 4
C_MSG: dict[str, tuple[int, int, int]] = {
    "CLAIM":      (0,   212, 255),
    "STRATEGY":   (220, 0,   255),
    "BOTTLENECK": (255, 200, 0),
    "COMPLETE":   (0,   255, 136),
    "INTENT":     (180, 180, 210),
}
C_MSG_DEFAULT = (140, 140, 160)

# ---------------------------------------------------------------------------
# Module-level animation state — Upgrade 3 & 5
# ---------------------------------------------------------------------------
_robot_visual_pos: dict[int, list[float]] = {}  # id -> [float_x, float_y]
_robot_trails:     dict[int, deque]       = {}  # id -> deque[(px, py)]
_TRAIL_LEN = 8
_LERP      = 0.22  # lerp fraction per frame at 60 fps

_prev_connection_status: str = "online"
_disconnect_flash_ticks: int = 0           # pygame ms when flash started
_FLASH_DURATION_MS = 500

# Button rects — updated each frame by draw_sidepanel
_BTN_DISCONNECT: Optional["pygame.Rect"] = None
_BTN_OPEN:       Optional["pygame.Rect"] = None
_BTN_BOTTLENECK: Optional["pygame.Rect"] = None
_BTN_RESET:      Optional["pygame.Rect"] = None
_BTN_SPEED:      Optional["pygame.Rect"] = None


def _ensure_pygame() -> None:
    """Raise ImportError if pygame is not installed."""
    if pygame is None:
        raise ImportError("pygame is required. Run: pip3 install pygame")


# ---------------------------------------------------------------------------
# Alpha drawing helper — used by trails and glow
# ---------------------------------------------------------------------------

def _draw_alpha_circle(
    surface: "pygame.Surface",
    color: tuple[int, int, int],
    center: tuple[int, int],
    radius: int,
    alpha: int,
) -> None:
    """Blit a circle with per-pixel alpha onto surface."""
    size = radius * 2 + 2
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.circle(s, (*color, alpha), (radius + 1, radius + 1), radius)
    surface.blit(s, (center[0] - radius - 1, center[1] - radius - 1))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_display(width: int = WINDOW_W, height: int = WINDOW_H) -> "pygame.Surface":
    """Initialise pygame and return the main screen surface."""
    _ensure_pygame()
    pygame.init()
    pygame.display.set_caption("FactoryMind R2D2 — Hack-a-Claw x NVIDIA")
    return pygame.display.set_mode((width, height))


def render(screen: "pygame.Surface", world_state: dict) -> None:
    """Draw one complete frame from world_state and flip the display."""
    global _prev_connection_status, _disconnect_flash_ticks
    _ensure_pygame()

    # Auto-detect connection_status transitions to trigger flash — Upgrade 5
    current = world_state.get("connection_status", "online")
    if current != _prev_connection_status:
        _disconnect_flash_ticks = pygame.time.get_ticks()
        _prev_connection_status = current

    screen.fill(C_BG)
    wall_set = {tuple(c) for c in world_state.get("wall", [])}

    draw_grid(screen)
    draw_walls(screen, wall_set)
    draw_workstations(screen, world_state.get("workstations", []))
    draw_robots(screen, world_state.get("robots", []))
    draw_sidepanel(screen, world_state)

    if current == "offline":
        draw_disconnect_banner(screen)

    _draw_flash_overlay(screen)
    pygame.display.flip()


# ---------------------------------------------------------------------------
# Grid — Upgrade 1
# ---------------------------------------------------------------------------

def draw_grid(screen: "pygame.Surface") -> None:
    """Blueprint grid: faint lines with brighter major lines every 5 cells."""
    from factorymind.state import GRID_WIDTH, GRID_HEIGHT
    for x in range(GRID_WIDTH + 1):
        color = C_GRID_MAJOR if x % 5 == 0 else C_GRID_FAINT
        px = GRID_OFFSET_X + x * CELL_SIZE
        pygame.draw.line(screen, color, (px, GRID_OFFSET_Y),
                         (px, GRID_OFFSET_Y + GRID_HEIGHT * CELL_SIZE))
    for y in range(GRID_HEIGHT + 1):
        color = C_GRID_MAJOR if y % 5 == 0 else C_GRID_FAINT
        py = GRID_OFFSET_Y + y * CELL_SIZE
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
# Workstations — Upgrade 2
# ---------------------------------------------------------------------------

_WS_ABBREV = {"Parts": "P", "Assembly": "A", "QA": "Q", "Shipping": "S"}

def draw_workstations(screen: "pygame.Surface", workstations: list[dict]) -> None:
    """Industrial equipment tiles: dark fill, 3D top-edge highlight, bold letter, pill label."""
    font_letter = pygame.font.SysFont("monospace", 28, bold=True)
    font_label  = pygame.font.SysFont("monospace", 11)

    for ws in workstations:
        x, y      = ws["pos"]
        color     = tuple(ws["color"])
        fill      = tuple(max(0, c - 120) for c in color)
        border    = tuple(max(0, c - 80)  for c in color)
        highlight = tuple(min(255, c + 60) for c in color)

        px = GRID_OFFSET_X + x * CELL_SIZE - CELL_SIZE
        py = GRID_OFFSET_Y + y * CELL_SIZE - CELL_SIZE
        w  = CELL_SIZE * 3
        h  = CELL_SIZE * 3
        rect = pygame.Rect(px, py, w, h)

        pygame.draw.rect(screen, fill,   rect, border_radius=4)
        pygame.draw.rect(screen, border, rect, 2, border_radius=4)

        # Inner top-edge highlight strip for 3D depth
        hl = pygame.Surface((w - 6, 3), pygame.SRCALPHA)
        hl.fill((*highlight, 110))
        screen.blit(hl, (px + 3, py + 3))

        # Bold single letter
        letter = _WS_ABBREV.get(ws["name"], ws["name"][0])
        ls = font_letter.render(letter, True, color)
        screen.blit(ls, (px + w // 2 - ls.get_width() // 2,
                         py + h // 2 - ls.get_height() // 2))

        # Dark pill label below
        lb   = font_label.render(ws["name"].upper(), True, (220, 220, 220))
        pw   = lb.get_width() + 10
        ph   = lb.get_height() + 4
        pill = pygame.Rect(px + w // 2 - pw // 2, py + h + 4, pw, ph)
        pygame.draw.rect(screen, (20, 20, 35), pill, border_radius=3)
        pygame.draw.rect(screen, border, pill, 1, border_radius=3)
        screen.blit(lb, (pill.x + 5, pill.y + 2))


# ---------------------------------------------------------------------------
# Robots + fading trails — Upgrade 3
# ---------------------------------------------------------------------------

def draw_robots(screen: "pygame.Surface", robots: list[dict]) -> None:
    """Smooth lerp movement with 8-position fading trails per robot."""
    from factorymind.state import LEADER
    font = pygame.font.SysFont("monospace", 9, bold=True)

    for robot in robots:
        rid  = robot["id"]
        role = robot["role"]
        gx, gy = robot["pos"]

        # Initialise visual state on first sight
        if rid not in _robot_visual_pos:
            _robot_visual_pos[rid] = [float(gx), float(gy)]
        if rid not in _robot_trails:
            _robot_trails[rid] = deque(maxlen=_TRAIL_LEN)

        # Lerp visual position toward actual grid position
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

        # Trail: oldest = dim/tiny → newest = bright/larger
        for i, (tx, ty) in enumerate(trail[:-1]):
            frac   = (i + 1) / n
            alpha  = int(15 + frac * 120)
            radius = max(1, int(1 + frac * 4))
            _draw_alpha_circle(screen, color, (tx, ty), radius, alpha)

        # Robot body + outer glow
        radius = CELL_SIZE // 2 - 1
        _draw_alpha_circle(screen, color, (px, py), radius + 3, 35)
        pygame.draw.circle(screen, color, (px, py), radius)
        pygame.draw.circle(screen, (0, 0, 0), (px, py), radius, 1)

        lbl = font.render(str(rid), True, (0, 0, 0))
        screen.blit(lbl, (px - lbl.get_width() // 2, py - lbl.get_height() // 2))


# ---------------------------------------------------------------------------
# Side panel — Upgrades 4 & 6
# ---------------------------------------------------------------------------

def draw_sidepanel(screen: "pygame.Surface", world_state: dict) -> None:
    """Ops-dashboard panel: stats tiles, agent reasoning feed, model badges, controls."""
    global _BTN_DISCONNECT, _BTN_OPEN, _BTN_BOTTLENECK, _BTN_RESET, _BTN_SPEED

    W   = screen.get_width() - PANEL_X
    H   = screen.get_height()
    PAD = 14
    x0  = PANEL_X + PAD

    offline = world_state.get("connection_status") == "offline"
    speed   = world_state.get("speed_multiplier", 1)

    pygame.draw.rect(screen, C_PANEL_BG,   (PANEL_X, 0, W, H))
    pygame.draw.line(screen, C_PANEL_EDGE, (PANEL_X, 0), (PANEL_X, H))

    f_title = pygame.font.SysFont("monospace", 20, bold=True)
    f_caps  = pygame.font.SysFont("monospace", 11)
    f_body  = pygame.font.SysFont("monospace", 12)
    f_big   = pygame.font.SysFont("monospace", 24, bold=True)
    f_small = pygame.font.SysFont("monospace", 10)
    f_btn   = pygame.font.SysFont("monospace", 12, bold=True)

    y = 18

    # --- Title + cyan underline ---
    ts = f_title.render("FACTORYMIND", True, C_ACCENT)
    screen.blit(ts, (x0, y))
    y += ts.get_height() + 2
    pygame.draw.line(screen, C_ACCENT, (x0, y), (PANEL_X + W - PAD, y), 1)
    y += 8

    # --- Layout subtitle ---
    sub = f_caps.render(world_state.get("layout", "UNKNOWN").replace("_", " "), True, C_DIM)
    screen.blit(sub, (x0, y))
    y += sub.get_height() + 6

    # --- Layout switcher buttons — Upgrade 6 ---
    active = world_state.get("layout", "")
    bw     = (W - PAD * 2 - 8) // 2
    _BTN_OPEN       = pygame.Rect(x0,          y, bw, 30)
    _BTN_BOTTLENECK = pygame.Rect(x0 + bw + 8, y, bw, 30)

    for btn, label, key in [
        (_BTN_OPEN,       "OPEN FLOOR", "OPEN_FLOOR"),
        (_BTN_BOTTLENECK, "BOTTLENECK", "BOTTLENECK_BRIDGE"),
    ]:
        is_active = (active == key)
        pygame.draw.rect(screen, (15, 25, 50) if is_active else (12, 15, 28), btn, border_radius=4)
        pygame.draw.rect(screen, C_ACCENT if is_active else (50, 50, 80),
                         btn, 2 if is_active else 1, border_radius=4)
        t = f_caps.render(label, True, C_ACCENT if is_active else C_DIM)
        screen.blit(t, (btn.x + btn.w // 2 - t.get_width() // 2,
                        btn.y + btn.h // 2 - t.get_height() // 2))
    y += 30 + 12

    # --- Stats tiles ---
    stats   = world_state.get("stats", {})
    elapsed = stats.get("elapsed", 0.0)
    tiles   = [
        ("COMPLETED", str(stats.get("completed", 0))),
        ("TIME",      f"{int(elapsed)//60:02d}:{int(elapsed)%60:02d}"),
        ("RATE/MIN",  f"{stats.get('rate', 0.0):.1f}"),
    ]
    tw = (W - PAD * 2 - 8) // 3
    th = 58
    for i, (lbl, val) in enumerate(tiles):
        tx        = x0 + i * (tw + 4)
        tile_rect = pygame.Rect(tx, y, tw, th)
        pygame.draw.rect(screen, (18, 18, 36), tile_rect, border_radius=4)
        pygame.draw.rect(screen, (40, 40, 70), tile_rect, 1, border_radius=4)
        ls = f_small.render(lbl, True, C_DIM)
        vs = f_big.render(val, True, C_ACCENT)
        screen.blit(ls, (tx + tw // 2 - ls.get_width() // 2, y + 6))
        screen.blit(vs, (tx + tw // 2 - vs.get_width() // 2, y + 22))
    y += th + 12

    # --- Divider ---
    pygame.draw.line(screen, C_PANEL_EDGE, (x0, y), (PANEL_X + W - PAD, y))
    y += 10

    # --- Agent Reasoning header ---
    ar = f_body.render("AGENT REASONING", True, C_ACCENT)
    screen.blit(ar, (x0, y))
    tk = f_small.render(f"tick {world_state.get('tick', 0)}  {speed}x", True, C_DIM)
    screen.blit(tk, (PANEL_X + W - PAD - tk.get_width(), y + 2))
    y += ar.get_height() + 6

    # --- Colour-coded message feed, newest first ---
    feed_bottom = H - 192
    line_h      = 15
    messages    = world_state.get("blackboard", [])[-20:][::-1]

    for msg in messages:
        if y + line_h > feed_bottom:
            break
        msg_type = msg.get("type", "")
        color    = C_MSG.get(msg_type, C_MSG_DEFAULT)
        ts_str   = time.strftime("%M:%S", time.localtime(msg.get("timestamp", 0.0)))
        from_id  = msg.get("from", "?")
        from_str = "★" if from_id == -1 else f"R{from_id}"
        content  = str(msg.get("content", ""))[:36]

        ts_s   = f_small.render(ts_str,              True, C_DIM)
        type_s = f_small.render(f"[{msg_type[:4]}]", True, color)
        from_s = f_small.render(from_str,             True, C_DIM)
        msg_s  = f_small.render(content,              True, (200, 200, 215))

        cx = x0
        screen.blit(ts_s,   (cx, y)); cx += ts_s.get_width()   + 4
        screen.blit(type_s, (cx, y)); cx += type_s.get_width() + 4
        screen.blit(from_s, (cx, y)); cx += from_s.get_width() + 4
        screen.blit(msg_s,  (cx, y))
        y += line_h

    # --- Bottom: model badges, branding, controls ---
    y = feed_bottom
    pygame.draw.line(screen, C_PANEL_EDGE, (x0, y), (PANEL_X + W - PAD, y))
    y += 8

    for role_str, model_str, color in [
        ("leaders:",    "Nemotron-Nano-9B",   C_ACCENT),
        ("strategist:", "Nemotron-Super-49B", (220, 0, 255)),
    ]:
        ls = f_small.render(f"  {role_str}", True, C_DIM)
        ms = f_small.render(f" {model_str}", True, color)
        screen.blit(ls, (x0, y))
        screen.blit(ms, (x0 + ls.get_width(), y))
        y += ls.get_height() + 2

    nv = f_small.render("Powered by NVIDIA NIM  |  ASUS Ascent GX10", True, (70, 70, 100))
    screen.blit(nv, (x0, y))
    y += nv.get_height() + 10

    # Control buttons: RESET | SPEED — preserve Person D's additions
    ctrl_bw    = (W - PAD * 2 - 8) // 2
    _BTN_RESET = pygame.Rect(x0,                y, ctrl_bw, 30)
    _BTN_SPEED = pygame.Rect(x0 + ctrl_bw + 8, y, ctrl_bw, 30)

    for btn, label in [(_BTN_RESET, "RESET"), (_BTN_SPEED, f"SPEED {speed}x →")]:
        pygame.draw.rect(screen, (18, 18, 36), btn, border_radius=4)
        pygame.draw.rect(screen, (50, 50, 80), btn, 1, border_radius=4)
        t = f_caps.render(label, True, C_DIM)
        screen.blit(t, (btn.x + btn.w // 2 - t.get_width() // 2,
                        btn.y + btn.h // 2 - t.get_height() // 2))
    y += 30 + 8

    # DISCONNECT / RECONNECT button — Upgrade 5
    ticks       = pygame.time.get_ticks()
    pulse       = 0.5 + 0.5 * math.sin(ticks / 350.0)
    _BTN_DISCONNECT = pygame.Rect(x0, y, W - PAD * 2, 44)

    if offline:
        border_color = (40, int(160 + pulse * 80), 60)
        pygame.draw.rect(screen, (10, 35, 18), _BTN_DISCONNECT, border_radius=6)
        pygame.draw.rect(screen, border_color,  _BTN_DISCONNECT, 2, border_radius=6)
        lbl = f_btn.render("↑  RECONNECT CLOUD", True, (60, 230, 100))
    else:
        br_r         = int(140 + pulse * 60)
        border_color = (br_r, 20, 20)
        pygame.draw.rect(screen, (40, 10, 10), _BTN_DISCONNECT, border_radius=6)
        pygame.draw.rect(screen, border_color,  _BTN_DISCONNECT, 2, border_radius=6)
        lbl = f_btn.render("⚡  DISCONNECT CLOUD", True, (220, 60, 60))

    screen.blit(lbl, (
        _BTN_DISCONNECT.x + _BTN_DISCONNECT.w // 2 - lbl.get_width() // 2,
        _BTN_DISCONNECT.y + _BTN_DISCONNECT.h // 2 - lbl.get_height() // 2,
    ))


# ---------------------------------------------------------------------------
# Disconnect banner — Upgrade 5
# ---------------------------------------------------------------------------

def draw_disconnect_banner(screen: "pygame.Surface") -> None:
    """Pulsing red banner across the top with GX10 green badges below."""
    ticks = pygame.time.get_ticks()
    pulse = 0.5 + 0.5 * math.sin(ticks / 280.0)
    bh    = 36
    r     = int(140 + pulse * 60)

    pygame.draw.rect(screen, (r, 12, 12), (0, 0, screen.get_width(), bh))

    f_banner = pygame.font.SysFont("monospace", 13, bold=True)
    msg  = "CLOUD DISCONNECTED  —  RUNNING LOCALLY ON ASUS ASCENT GX10"
    surf = f_banner.render(msg, True, (255, 220, 220))
    screen.blit(surf, (screen.get_width() // 2 - surf.get_width() // 2, 8))

    # Green badges row beneath the banner
    f_badge = pygame.font.SysFont("monospace", 11)
    bx      = screen.get_width() // 2 - 220
    by      = bh + 4
    for text in ["✓ Nemotron-9B on GX10", "✓ Nemotron-49B on GX10"]:
        bs    = f_badge.render(text, True, (80, 230, 110))
        badge = pygame.Rect(bx, by, bs.get_width() + 14, bs.get_height() + 6)
        pygame.draw.rect(screen, C_BADGE_OK,    badge, border_radius=3)
        pygame.draw.rect(screen, (40, 140, 60), badge, 1, border_radius=3)
        screen.blit(bs, (bx + 7, by + 3))
        bx += badge.w + 12


def _draw_flash_overlay(screen: "pygame.Surface") -> None:
    """Brief red flash on disconnect/reconnect transition — fades over 500 ms."""
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
# Button hit detection
# ---------------------------------------------------------------------------

def get_button_click(pos: tuple[int, int]) -> Optional[str]:
    """Return which button was clicked, or None.

    Possible returns: "disconnect", "layout_open", "layout_bottleneck",
                      "reset", "speedup"
    """
    if _BTN_DISCONNECT and _BTN_DISCONNECT.collidepoint(pos):
        return "disconnect"
    if _BTN_OPEN and _BTN_OPEN.collidepoint(pos):
        return "layout_open"
    if _BTN_BOTTLENECK and _BTN_BOTTLENECK.collidepoint(pos):
        return "layout_bottleneck"
    if _BTN_RESET and _BTN_RESET.collidepoint(pos):
        return "reset"
    if _BTN_SPEED and _BTN_SPEED.collidepoint(pos):
        return "speedup"
    return None


# ---------------------------------------------------------------------------
# Standalone smoke test — run: python3 -m factorymind.render
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from factorymind.state import create_initial_state, OPEN_FLOOR

    _ensure_pygame()

    fake_state = create_initial_state(OPEN_FLOOR)
    fake_state["blackboard"] = [
        {"from": 0,  "type": "CLAIM",      "content": "Claiming Parts->Assembly task 3; 4 cells to pickup.", "timestamp": time.time() - 20},
        {"from": 1,  "type": "INTENT",     "content": "Moving to Assembly station",                          "timestamp": time.time() - 15},
        {"from": -1, "type": "STRATEGY",   "content": "Prioritise QA queue — route workers south.",          "timestamp": time.time() - 10},
        {"from": 2,  "type": "COMPLETE",   "content": "task-1 delivered to Shipping",                        "timestamp": time.time() - 5},
        {"from": 3,  "type": "BOTTLENECK", "content": "Bridge congestion at y=25 detected",                  "timestamp": time.time() - 2},
        {"from": 4,  "type": "CLAIM",      "content": "W4 grabbing task 7 (Parts->QA); 3 cells to pickup.",  "timestamp": time.time() - 1},
    ]
    fake_state["stats"]            = {"completed": 7, "elapsed": 82.0, "rate": 5.1}
    fake_state["speed_multiplier"] = 2

    screen = init_display()
    clock  = pygame.time.Clock()
    end    = time.time() + 8.0

    while time.time() < end:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                action = get_button_click(event.pos)
                if action == "disconnect":
                    s = fake_state["connection_status"]
                    fake_state["connection_status"] = "offline" if s == "online" else "online"
        render(screen, fake_state)
        clock.tick(60)

    pygame.quit()
    print("render.py v2 smoke test OK")
