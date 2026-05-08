"""Disruption handling and route re-planning - Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.grid_model import Grid
from src.astar_planner import astar


_LEG_NAMES = ["hub -> pickup", "pickup -> drop-off", "drop-off -> hub"]


@dataclass
class ActiveDelivery:
    job_id: str
    drone_id: str
    drone_type: str
    max_range: int
    waypoints: list[tuple[int, int]]   # [hub, pickup, dropoff, hub]
    next_wp_index: int                 # index of next target in waypoints
    current_pos: tuple[int, int]       # where the drone physically is
    segments: list[tuple[str, list[tuple[int, int]]]]  # (leg_name, path)
    status: str = "in_flight"          # in_flight | rerouted | failed


def make_active_delivery(
    assignment: dict,
    drone_range: int,
    *,
    advance_to_pickup: bool = True,
) -> ActiveDelivery:
    """
    Build an ActiveDelivery from a completed assignment dict.
    advance_to_pickup=True simulates the drone having already reached
    the pickup point, so only legs 2 and 3 remain - making the disruption
    demo more realistic.
    """
    hub = assignment["hub"]
    pickup = assignment["pickup"]
    dropoff = assignment["dropoff"]
    waypoints = [hub, pickup, dropoff, hub]

    all_segs = [(name, res.path or []) for name, res in assignment["segments"]]

    if advance_to_pickup and len(all_segs) > 1:
        return ActiveDelivery(
            job_id=assignment["job"],
            drone_id=assignment["drone"],
            drone_type=assignment["type"],
            max_range=drone_range,
            waypoints=waypoints,
            next_wp_index=2,        # next target is dropoff
            current_pos=pickup,
            segments=all_segs[1:],  # legs from pickup onward
        )

    return ActiveDelivery(
        job_id=assignment["job"],
        drone_id=assignment["drone"],
        drone_type=assignment["type"],
        max_range=drone_range,
        waypoints=waypoints,
        next_wp_index=1,
        current_pos=hub,
        segments=all_segs,
    )


def _segments_contain(
    segments: list[tuple[str, list[tuple[int, int]]]],
    cell: tuple[int, int],
) -> bool:
    return any(cell in path for _, path in segments)


def _pick_disruption_cell(delivery: ActiveDelivery) -> Optional[tuple[int, int]]:
    """Return a non-waypoint cell from the middle of the longest remaining segment."""
    protected = set(delivery.waypoints)
    best = max(delivery.segments, key=lambda s: len(s[1]), default=None)
    if not best:
        return None
    _, path = best
    mid = len(path) // 2
    for i in list(range(mid, len(path))) + list(range(mid - 1, -1, -1)):
        if path[i] not in protected:
            return path[i]
    return None


def _reroute(grid: Grid, delivery: ActiveDelivery) -> tuple[bool, str]:
    """Re-plan all remaining legs from current_pos. Updates delivery.segments in place."""
    remaining_wps = delivery.waypoints[delivery.next_wp_index:]
    label_offset = delivery.next_wp_index - 1
    new_segs: list[tuple[str, list[tuple[int, int]]]] = []
    pos = delivery.current_pos
    total_steps = 0

    for i, wp in enumerate(remaining_wps):
        idx = label_offset + i
        label = _LEG_NAMES[idx] if idx < len(_LEG_NAMES) else f"leg {idx + 1}"
        res = astar(pos, wp, grid)
        if res.error:
            return False, f"No path to {wp}: {res.error}"
        path = res.path or []
        total_steps += len(path) - 1
        if total_steps > delivery.max_range:
            return False, (
                f"Rerouted path ({total_steps} steps) exceeds "
                f"drone range ({delivery.max_range} cells)."
            )
        new_segs.append((label, path))
        pos = wp

    delivery.segments = new_segs
    delivery.status = "rerouted"
    return True, f"Rerouted successfully ({total_steps} steps remaining)."


def activate_disruption(
    grid: Grid,
    cell: tuple[int, int],
    active_deliveries: list[ActiveDelivery],
    step: int = 0,
) -> list[str]:
    """
    Mark cell as no-fly. Find all in-flight deliveries whose remaining path
    crosses that cell and reroute them. Returns event log lines.
    """
    logs: list[str] = []
    r, c = cell
    grid.grid[r][c].no_fly = True
    logs.append(f"Step {step}: no-fly cell activated at {cell}.")

    affected = 0
    for d in active_deliveries:
        if d.status != "in_flight":
            continue
        if not _segments_contain(d.segments, cell):
            logs.append(f"  {d.drone_id} ({d.job_id}): route unaffected.")
            continue
        affected += 1
        logs.append(
            f"  {d.drone_id} ({d.job_id}): route crosses {cell} - "
            f"rerouting from {d.current_pos}."
        )
        ok, msg = _reroute(grid, d)
        if ok:
            logs.append(f"  {d.drone_id} rerouted using A*. {msg}")
        else:
            d.status = "failed"
            logs.append(f"  {d.drone_id} cannot reach destination safely. {msg}")

    if affected == 0:
        logs.append("  No active routes were affected by this disruption.")

    return logs


def run_disruption_demo(
    grid: Grid,
    assignments: list[dict],
    drone_slots: list[dict],
    step: int = 11,
) -> Optional[ActiveDelivery]:
    """
    Phase 4 demo: take the first successful delivery, advance the drone to its
    pickup point, inject a no-fly disruption on the remaining path, reroute
    using A*, then display a before/after plot.
    """
    from src.visualization import plot_disruption_reroute  # late import avoids circular

    first_ok = next((a for a in assignments if a["status"] == "OK"), None)
    if not first_ok:
        print("[Phase 4] No successful deliveries available for disruption demo.")
        return None

    drone = next(
        (s for s in drone_slots if s["id"] == first_ok["drone"]),
        drone_slots[0] if drone_slots else {"range": 20},
    )
    active = make_active_delivery(first_ok, drone["range"], advance_to_pickup=True)
    original_segs = [(name, list(path)) for name, path in active.segments]

    disruption_cell = _pick_disruption_cell(active)
    if not disruption_cell:
        print("[Phase 4] Could not find a suitable disruption cell on the route.")
        return None

    print(f"\n{'=' * 60}")
    print(" PHASE 4 - DISRUPTION & REROUTING")
    print(f"{'=' * 60}")
    print(
        f"  Drone : {active.drone_id} ({active.drone_type}, "
        f"range={active.max_range} cells)"
    )
    print(f"  Status: in flight - arrived at pickup {active.current_pos}")
    print(
        f"  Next  : drop-off {active.waypoints[2]}  ->  hub {active.waypoints[0]}"
    )
    print("\n  Original remaining path:")
    for name, path in original_segs:
        print(f"    {name}: {max(0, len(path) - 1)} steps")

    print()
    logs = activate_disruption(grid, disruption_cell, [active], step=step)
    for log in logs:
        print(log)

    if active.status == "rerouted":
        print("\n  Updated remaining path:")
        for name, path in active.segments:
            print(f"    {name}: {max(0, len(path) - 1)} steps")
    elif active.status == "failed":
        print("\n  Delivery marked as FAILED - no safe route exists.")

    outcome = "rerouted" if active.status == "rerouted" else "FAILED - no safe path"
    plot_disruption_reroute(
        grid,
        original_segs,
        active.segments if active.status == "rerouted" else [],
        hub=active.waypoints[0],
        pickup=active.waypoints[1],
        dropoff=active.waypoints[2],
        disrupted_cell=disruption_cell,
        title=f"4. Disruption at {disruption_cell}  |  {active.drone_id} {outcome}",
        show=True,
    )

    return active
