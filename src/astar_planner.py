"""
A* path planning on the grid (4-neighbor, Manhattan heuristic).
Step cost: 1.0 default, 0.8 on commercial. Blocked: no_fly.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappush, heappop
from pathlib import Path
import sys
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.grid_model import Grid, Zone


@dataclass
class AStarResult:
    path: Optional[list[tuple[int, int]]]
    total_cost: float
    error: Optional[str] = None


def _in_bounds(grid: Grid, r: int, c: int) -> bool:
    return 0 <= r < grid.rows and 0 <= c < grid.cols


def manhattan_heuristic(r1: int, c1: int, r2: int, c2: int) -> int:
    return abs(r1 - r2) + abs(c1 - c2)


def step_cost_into(cell) -> float:
    return 0.8 if cell.zone == Zone.COMMERCIAL else 1.0


def astar(
    start: tuple[int, int],
    goal: tuple[int, int],
    grid: Grid,
) -> AStarResult:
    """
    start, goal: (row, col). Returns path as list of (row,col), total cost, or error.
    """
    sr, sc = start
    gr, gc = goal

    if not _in_bounds(grid, sr, sc) or not _in_bounds(grid, gr, gc):
        return AStarResult(None, 0.0, "Start or goal out of bounds.")

    if grid.grid[sr][sc].no_fly or grid.grid[gr][gc].no_fly:
        return AStarResult(None, 0.0, "Start or goal is in a no-fly cell.")

    if (sr, sc) == (gr, gc):
        return AStarResult([(sr, sc)], 0.0, None)

    start_k = (sr, sc)
    goal_k = (gr, gc)

    # priority: f = g + h; tie-break counter keeps heap order stable
    counter = 0
    open_heap: list[tuple[float, int, int, int]] = []
    h0 = manhattan_heuristic(sr, sc, gr, gc)
    heappush(open_heap, (h0, 0.0, counter, sr, sc))
    counter += 1

    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start_k: 0.0}

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    goal_found = False

    while open_heap:
        _f, g_pop, _, r, c = heappop(open_heap)
        current = (r, c)
        if g_pop != g_score[current]:
            continue
        if current == goal_k:
            goal_found = True
            break
        gcost = g_pop

        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if not _in_bounds(grid, nr, nc):
                continue
            ncell = grid.grid[nr][nc]
            if ncell.no_fly:
                continue
            nxt = (nr, nc)
            tentative = gcost + step_cost_into(ncell)
            if tentative < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative
                f = tentative + manhattan_heuristic(nr, nc, gr, gc)
                heappush(open_heap, (f, tentative, counter, nr, nc))
                counter += 1

    if not goal_found:
        return AStarResult(None, 0.0, "No path exists (goal unreachable).")

    # reconstruct
    path_rev: list[tuple[int, int]] = [goal_k]
    cur = goal_k
    while cur != start_k:
        cur = came_from[cur]
        path_rev.append(cur)
    path_rev.reverse()
    return AStarResult(path_rev, g_score[goal_k], None)


def plan_delivery_segments(
    grid: Grid,
    hub: tuple[int, int],
    pickup: tuple[int, int],
    dropoff: tuple[int, int],
) -> tuple[list[tuple[str, AStarResult]], Optional[str]]:
    """
    hub -> pickup -> dropoff -> hub. Returns list of (segment name, result) and optional error.
    """
    legs = [
        ("hub -> pickup", hub, pickup),
        ("pickup -> drop-off", pickup, dropoff),
        ("drop-off -> hub", dropoff, hub),
    ]
    out: list[tuple[str, AStarResult]] = []
    for name, a, b in legs:
        res = astar(a, b, grid)
        if res.error:
            return out, f"{name}: {res.error}"
        out.append((name, res))
    return out, None


# --- waypoint helpers for demos ---


def first_hub(grid: Grid) -> Optional[tuple[int, int]]:
    for row in grid.grid:
        for cell in row:
            if cell.is_hub:
                return (cell.row, cell.col)
    return None


def first_commercial(grid: Grid) -> Optional[tuple[int, int]]:
    for row in grid.grid:
        for cell in row:
            if cell.zone == Zone.COMMERCIAL:
                return (cell.row, cell.col)
    return None


def pick_delivery_waypoints(grid: Grid) -> Optional[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    hubs = [(c.row, c.col) for row in grid.grid for c in row if c.is_hub]
    pickups_set: set[tuple[int, int]] = set()
    for row in grid.grid:
        for cell in row:
            if cell.is_medical_pickup:
                pickups_set.add((cell.row, cell.col))
    for row in grid.grid:
        for cell in row:
            if cell.zone == Zone.COMMERCIAL:
                pickups_set.add((cell.row, cell.col))
    pickups = list(pickups_set)
    dropoffs = []
    for row in grid.grid:
        for cell in row:
            if cell.zone == Zone.RESIDENTIAL:
                dropoffs.append((cell.row, cell.col))

    def md(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    best = None
    best_key = -1
    for h in hubs:
        for p in pickups:
            if p == h:
                continue
            for d in dropoffs:
                if d == h or d == p:
                    continue
                spread = md(h, p) + md(p, d)
                if spread > best_key:
                    best_key = spread
                    best = (h, p, d)

    if best is not None:
        return best

    hub = first_hub(grid)
    if not hub:
        return None
    pickup = None
    for row in grid.grid:
        for cell in row:
            if cell.is_medical_pickup and (cell.row, cell.col) != hub:
                pickup = (cell.row, cell.col)
                break
        if pickup:
            break
    if not pickup:
        pickup = first_commercial(grid)
        if pickup == hub:
            pickup = None
            for row in grid.grid:
                for cell in row:
                    if cell.zone == Zone.COMMERCIAL and (cell.row, cell.col) != hub:
                        pickup = (cell.row, cell.col)
                        break
                if pickup:
                    break
    dropoff = None
    for row in grid.grid:
        for cell in row:
            if cell.zone == Zone.RESIDENTIAL and cell.demand > 0:
                if (cell.row, cell.col) not in (hub, pickup):
                    dropoff = (cell.row, cell.col)
                    break
        if dropoff:
            break
    if not dropoff:
        for row in grid.grid:
            for cell in row:
                if cell.zone == Zone.RESIDENTIAL and (cell.row, cell.col) not in (
                    hub,
                    pickup,
                ):
                    dropoff = (cell.row, cell.col)
                    break
            if dropoff:
                break
    if pickup and dropoff:
        return hub, pickup, dropoff
    return None


def clear_no_fly_at(grid: Grid, points: list[tuple[int, int]]) -> None:
    for r, c in points:
        if _in_bounds(grid, r, c):
            grid.grid[r][c].no_fly = False

