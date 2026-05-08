from pathlib import Path
import random
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.grid_model import Grid, Zone
from src.layout_validator import print_validation_report, repair_layout
from src.fleet_selector import format_fleet_summary, select_fleet_ga
from src.astar_planner import (
    clear_no_fly_at,
    pick_delivery_waypoints,
    plan_delivery_segments,
)
from src.visualization import plot_delivery_routes, plot_zone_layout


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

    wps = pick_delivery_waypoints(grid)
    if not wps:
        print("\n[A*] No hub / pickup / drop-off — skipping routing.")
        plot_zone_layout(grid, title="1. Zone layout (after CSP repair)", show=True)
        return grid

    hub, pickup, dropoff = wps
    reserved = {hub, pickup, dropoff}
    clear_no_fly_at(grid, list(reserved))
    sprinkle_no_fly(grid, n=12, seed=seed, avoid=reserved)
    clear_no_fly_at(grid, list(reserved))

    plot_zone_layout(
        grid,
        title="1. Zone map (//// = no-fly)",
        show=True,
        mark_no_fly=True,
    )

    print(f"\n[A*] Waypoints: hub={hub}, pickup={pickup}, drop-off={dropoff}")
    segments, err = plan_delivery_segments(grid, hub, pickup, dropoff)
    if err:
        print(f"[A*] FAILED: {err}")
        plot_zone_layout(
            grid,
            title="2. Zone layout (routing failed)",
            show=True,
            mark_no_fly=True,
        )
        return grid

    for name, res in segments:
        print(f"  {name}: cost={res.total_cost:.2f}  {_path_pretty(res.path or [])}")

    plot_delivery_routes(
        grid,
        [(name, res.path or []) for name, res in segments],
        hub=hub,
        pickup=pickup,
        dropoff=dropoff,
        title="2. A* delivery: hub -> pickup -> drop-off -> hub",
        show=True,
    )
    return grid


def run_demo() -> Grid:
    return run_full_demo()


def main() -> None:
    run_full_demo()


if __name__ == "__main__":
    main()
