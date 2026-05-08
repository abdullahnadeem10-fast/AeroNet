from pathlib import Path
import random
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.grid_model import Grid, Zone
from src.layout_validator import print_validation_report, repair_layout
from src.fleet_selector import (
    HEAVY_RANGE_CELLS,
    LIGHT_RANGE_CELLS,
    FleetResult,
    format_fleet_summary,
    select_fleet_ga,
)
from src.astar_planner import (
    clear_no_fly_at,
    pick_multiple_delivery_waypoints,
    plan_delivery_segments,
)
from src.visualization import plot_delivery_routes, plot_demand_heatmap, plot_zone_layout
from src.disruption_handler import run_disruption_demo


def sprinkle_no_fly(
    grid: Grid,
    n: int = 8,
    seed: int = 1,
    avoid: set[tuple[int, int]] | None = None,
) -> None:
    """Mark random cells as no-fly (not hubs, not reserved waypoints)."""
    avoid = avoid or set()
    rnd = random.Random(seed)
    placed = 0
    for _ in range(600):
        if placed >= n:
            break
        r, c = rnd.randrange(grid.rows), rnd.randrange(grid.cols)
        if (r, c) in avoid:
            continue
        if grid.grid[r][c].is_hub:
            continue
        grid.grid[r][c].no_fly = True
        placed += 1


def _build_drone_slots(fleet: FleetResult) -> list[dict]:
    slots = []
    for i in range(fleet.light_count):
        slots.append({"id": f"D{i + 1}", "type": "Light", "range": LIGHT_RANGE_CELLS})
    for j in range(fleet.heavy_count):
        slots.append({"id": f"D{fleet.light_count + j + 1}", "type": "Heavy", "range": HEAVY_RANGE_CELLS})
    return slots


def _assign_deliveries(
    grid,
    deliveries: list,
    drone_slots: list[dict],
) -> list[dict]:
    results = []
    n = len(drone_slots)
    for i, (hub, pickup, dropoff) in enumerate(deliveries):
        assigned = False
        for offset in range(n):
            drone = drone_slots[(i + offset) % n]
            segs, err = plan_delivery_segments(
                grid, hub, pickup, dropoff, max_range=drone["range"]
            )
            if not err:
                steps = sum(len(r.path) - 1 for _, r in segs if r.path)
                cost = sum(r.total_cost for _, r in segs)
                results.append({
                    "job": f"J{i + 1}",
                    "drone": drone["id"],
                    "type": drone["type"],
                    "hub": hub,
                    "pickup": pickup,
                    "dropoff": dropoff,
                    "steps": steps,
                    "cost": cost,
                    "status": "OK",
                    "segments": segs,
                })
                assigned = True
                break
        if not assigned:
            results.append({
                "job": f"J{i + 1}",
                "drone": "--",
                "type": "--",
                "hub": hub,
                "pickup": pickup,
                "dropoff": dropoff,
                "steps": 0,
                "cost": 0.0,
                "status": "FAILED",
                "segments": [],
            })
    return results


def _print_delivery_table(assignments: list[dict]) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(" DELIVERY ASSIGNMENT TABLE")
    print(sep)
    print(f"{'Job':<5} {'Drone':<6} {'Type':<7} {'Hub':<8} {'Pickup':<8} {'Drop-off':<9} {'Steps':>5} {'Cost':>6}  Status")
    print("-" * 70)
    for a in assignments:
        steps_s = str(a["steps"]) if a["status"] == "OK" else "--"
        cost_s = f"{a['cost']:.1f}" if a["status"] == "OK" else "--"
        print(
            f"{a['job']:<5} {a['drone']:<6} {a['type']:<7} "
            f"{str(a['hub']):<8} {str(a['pickup']):<8} {str(a['dropoff']):<9} "
            f"{steps_s:>5} {cost_s:>6}  {a['status']}"
        )
    ok = sum(1 for a in assignments if a["status"] == "OK")
    print("-" * 70)
    print(f"  {ok} assigned   {len(assignments) - ok} failed\n")


def _path_pretty(path: list[tuple[int, int]]) -> str:
    if len(path) <= 10:
        return str(path)
    return f"{path[:4]} ... {path[-4:]}  ({len(path)} cells)"


def run_full_demo(seed: int = 42) -> Grid:
    random.seed(seed)
    grid = Grid(
        10,
        10,
        zone_limits={
            Zone.RESIDENTIAL: 10,
            Zone.COMMERCIAL: 10,
            Zone.HOSPITAL: 3,
            Zone.SCHOOL: 5,
            Zone.INDUSTRIAL: 8,
        },
    )
    grid.populate_grid()
    print_validation_report(grid)
    repair_layout(grid)
    print_validation_report(grid)

    fleet = select_fleet_ga(grid, budget=15_000, seed=seed)
    print()
    print(format_fleet_summary(fleet))

    deliveries = pick_multiple_delivery_waypoints(grid, n=8)
    if not deliveries:
        print("\n[A*] No deliveries could be generated - skipping routing.")
        plot_zone_layout(grid, title="1. Zone layout (after CSP repair)", show=True)
        return grid

    reserved: set[tuple[int, int]] = set()
    for hub, pickup, dropoff in deliveries:
        reserved.update([hub, pickup, dropoff])
    clear_no_fly_at(grid, list(reserved))
    sprinkle_no_fly(grid, n=12, seed=seed, avoid=reserved)
    clear_no_fly_at(grid, list(reserved))

    drone_slots = _build_drone_slots(fleet)
    assignments = _assign_deliveries(grid, deliveries, drone_slots)
    _print_delivery_table(assignments)

    plot_zone_layout(
        grid,
        title="1. Zone map (//// = no-fly)",
        show=True,
        mark_no_fly=True,
    )
    plot_demand_heatmap(grid, title="2. Delivery demand heatmap", show=True)

    first_ok = next((a for a in assignments if a["status"] == "OK"), None)
    if first_ok:
        print(
            f"[Route] {first_ok['job']} assigned to "
            f"{first_ok['drone']} ({first_ok['type']}, "
            f"range={LIGHT_RANGE_CELLS if first_ok['type'] == 'Light' else HEAVY_RANGE_CELLS} cells)"
        )
        for name, res in first_ok["segments"]:
            print(f"  {name}: cost={res.total_cost:.2f}  {_path_pretty(res.path or [])}")
        plot_delivery_routes(
            grid,
            [(name, res.path or []) for name, res in first_ok["segments"]],
            hub=first_ok["hub"],
            pickup=first_ok["pickup"],
            dropoff=first_ok["dropoff"],
            title=f"3. A* delivery: {first_ok['job']} - hub -> pickup -> drop-off -> hub",
            show=True,
        )

    run_disruption_demo(grid, assignments, drone_slots)

    return grid


def run_demo() -> Grid:
    return run_full_demo()


def main() -> None:
    run_full_demo()


if __name__ == "__main__":
    main()
