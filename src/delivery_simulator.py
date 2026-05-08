"""Full 20-step delivery simulation orchestration -- Phase 6."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.grid_model import Grid, Zone
from src.layout_validator import validate_layout, repair_layout
from src.fleet_selector import (
    select_fleet_ga,
    LIGHT_RANGE_CELLS,
    HEAVY_RANGE_CELLS,
)
from src.astar_planner import (
    pick_multiple_delivery_waypoints,
    plan_delivery_segments,
    astar,
    clear_no_fly_at,
)
from src.ml_pipeline import (
    generate_demand_dataset,
    train_demand_models,
    demand_forecast_for_grid,
    generate_telemetry_dataset,
    train_anomaly_models,
    ANOMALY_LABELS,
)

LOG_PATH = _ROOT / "report" / "simulation_log.txt"


# ─────────────────────────────────────────────────────────────
# DATA TYPES
# ─────────────────────────────────────────────────────────────

@dataclass
class DroneState:
    job_id: str
    drone_id: str
    drone_type: str
    max_range: int
    hub: tuple[int, int]
    pickup: tuple[int, int]
    dropoff: tuple[int, int]
    path: list[tuple[int, int]]   # flat ordered cell list for the full mission
    step_index: int = 0            # index of current position in path
    status: str = "in_flight"      # in_flight | returning | completed | returned | failed
    anomaly: Optional[str] = None

    @property
    def current_pos(self) -> tuple[int, int]:
        idx = min(self.step_index, len(self.path) - 1)
        return self.path[idx] if self.path else self.hub

    @property
    def progress_pct(self) -> int:
        if len(self.path) <= 1:
            return 100
        return int(100 * self.step_index / (len(self.path) - 1))


# ─────────────────────────────────────────────────────────────
# INLINED HELPERS (avoid circular import with main.py)
# ─────────────────────────────────────────────────────────────

def _build_drone_slots(fleet) -> list[dict]:
    slots = []
    for i in range(fleet.light_count):
        slots.append({"id": f"D{i + 1}", "type": "Light", "range": LIGHT_RANGE_CELLS})
    for j in range(fleet.heavy_count):
        slots.append({"id": f"D{fleet.light_count + j + 1}", "type": "Heavy", "range": HEAVY_RANGE_CELLS})
    return slots


def _sprinkle_no_fly(
    grid: Grid, n: int = 10, seed: int = 1, avoid: set | None = None
) -> None:
    avoid = avoid or set()
    rnd = random.Random(seed)
    placed = 0
    for _ in range(600):
        if placed >= n:
            break
        r, c = rnd.randrange(grid.rows), rnd.randrange(grid.cols)
        if (r, c) in avoid or grid.grid[r][c].is_hub:
            continue
        grid.grid[r][c].no_fly = True
        placed += 1


def _assign_deliveries(
    grid: Grid, deliveries: list, drone_slots: list[dict]
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
                results.append({
                    "job": f"J{i + 1}", "drone": drone["id"], "type": drone["type"],
                    "hub": hub, "pickup": pickup, "dropoff": dropoff,
                    "steps": sum(len(r.path) - 1 for _, r in segs if r.path),
                    "status": "OK", "segments": segs,
                })
                assigned = True
                break
        if not assigned:
            results.append({
                "job": f"J{i + 1}", "drone": "--", "type": "--",
                "hub": hub, "pickup": pickup, "dropoff": dropoff,
                "steps": 0, "status": "FAILED", "segments": [],
            })
    return results


# ─────────────────────────────────────────────────────────────
# DRONE MOVEMENT & PATH HELPERS
# ─────────────────────────────────────────────────────────────

def _flatten_segments(
    segments: list[tuple[str, list[tuple[int, int]]]]
) -> list[tuple[int, int]]:
    """Merge segment paths into one flat list, no duplicated waypoints."""
    flat: list[tuple[int, int]] = []
    for i, (_, path) in enumerate(segments):
        flat.extend(path if i == 0 else path[1:])
    return flat


def _make_drone_state(assignment: dict, drone: dict) -> Optional[DroneState]:
    segs = [(name, res.path or []) for name, res in assignment["segments"]]
    path = _flatten_segments(segs)
    if not path:
        return None
    return DroneState(
        job_id=assignment["job"],
        drone_id=assignment["drone"],
        drone_type=assignment["type"],
        max_range=drone["range"],
        hub=assignment["hub"],
        pickup=assignment["pickup"],
        dropoff=assignment["dropoff"],
        path=path,
    )


def _advance_drones(
    drones: list[DroneState], step: int, cells: int = 2
) -> list[str]:
    """
    Move every active drone forward by `cells` positions per simulation step.
    Returns notable event lines (completions / returns).
    """
    logs: list[str] = []
    for d in drones:
        if d.status not in ("in_flight", "returning"):
            continue
        for _ in range(cells):
            if d.step_index < len(d.path) - 1:
                d.step_index += 1
        if d.step_index >= len(d.path) - 1:
            if d.status == "in_flight":
                d.status = "completed"
                logs.append(
                    f"  {d.drone_id} ({d.job_id}) completed delivery at step {step}."
                )
            elif d.status == "returning":
                d.status = "returned"
                logs.append(
                    f"  {d.drone_id} ({d.job_id}) returned to hub (delivery delayed) at step {step}."
                )
    return logs


def _remaining_waypoints(drone: DroneState) -> list[tuple[int, int]]:
    """Return which of pickup / dropoff / hub still appear in the remaining path."""
    remaining_set = set(drone.path[drone.step_index:])
    wps = []
    for wp in [drone.pickup, drone.dropoff, drone.hub]:
        if wp in remaining_set:
            wps.append(wp)
    return wps if wps else [drone.hub]


def _replan_path(
    grid: Grid,
    start: tuple[int, int],
    waypoints: list[tuple[int, int]],
    max_range: int,
) -> Optional[list[tuple[int, int]]]:
    """A* through each waypoint in order. Returns flat path or None if impossible."""
    flat = [start]
    pos = start
    total = 0
    for wp in waypoints:
        res = astar(pos, wp, grid)
        if res.error or not res.path:
            return None
        total += len(res.path) - 1
        if total > max_range:
            return None
        flat.extend(res.path[1:])
        pos = wp
    return flat


def _pick_disruption_cell(drones: list[DroneState]) -> Optional[tuple[int, int]]:
    """Find a non-waypoint cell that is still ahead of some in-flight drone."""
    for d in drones:
        if d.status != "in_flight":
            continue
        protected = {d.hub, d.pickup, d.dropoff}
        ahead = d.path[d.step_index + 2:]   # skip current + immediate next
        for cell in ahead:
            if cell not in protected:
                return cell
    return None


# ─────────────────────────────────────────────────────────────
# DISRUPTION & ANOMALY
# ─────────────────────────────────────────────────────────────

def _disrupt_and_reroute(
    grid: Grid,
    drones: list[DroneState],
    cell: tuple[int, int],
    step: int,
) -> list[str]:
    logs: list[str] = []
    r, c = cell
    grid.grid[r][c].no_fly = True
    logs.append(f"Step {step}: no-fly cell activated at {cell}.")

    affected = 0
    for d in drones:
        if d.status != "in_flight":
            continue
        if cell not in set(d.path[d.step_index + 1:]):
            continue
        affected += 1
        logs.append(
            f"  {d.drone_id} ({d.job_id}): route crosses {cell} - "
            f"rerouting from {d.current_pos}."
        )
        wps = _remaining_waypoints(d)
        new_suffix = _replan_path(grid, d.current_pos, wps, d.max_range)
        if new_suffix:
            d.path = d.path[: d.step_index + 1] + new_suffix[1:]
            logs.append(
                f"  {d.drone_id} rerouted using A*. "
                f"{len(new_suffix) - 1} steps remaining."
            )
        else:
            d.status = "failed"
            logs.append(
                f"  {d.drone_id} cannot reach destination safely. Delivery failed."
            )

    if affected == 0:
        logs.append("  No active routes affected by this disruption.")
    return logs


def _inject_anomaly(
    grid: Grid,
    drones: list[DroneState],
    anomaly_model,
    step: int,
) -> list[str]:
    logs: list[str] = []
    candidates = [d for d in drones if d.status == "in_flight"]
    if not candidates:
        logs.append(f"Step {step}: Anomaly classifier active — no drones in flight to affect.")
        return logs

    # Target the drone furthest along its route
    target = max(candidates, key=lambda d: d.step_index)

    # Battery-anomaly telemetry signature (as defined in generate_telemetry_dataset)
    x = np.array([[9.5, 10.1, 0.6, 0.0, 0.1]])
    pred_label = int(anomaly_model.predict(x)[0])
    anomaly_name = ANOMALY_LABELS.get(pred_label, "Unknown")
    target.anomaly = anomaly_name

    logs.append(f"Step {step}: {anomaly_name} detected for {target.drone_id} at {target.current_pos}.")

    res = astar(target.current_pos, target.hub, grid)
    if not res.error and res.path:
        target.path = target.path[: target.step_index + 1] + res.path[1:]
        target.status = "returning"
        logs.append(
            f"  {target.drone_id} forced to return to hub {target.hub}. "
            f"Delivery marked as delayed."
        )
    else:
        target.status = "failed"
        logs.append(f"  {target.drone_id} cannot return to hub. Delivery failed.")
    return logs


# ─────────────────────────────────────────────────────────────
# MAIN 20-STEP RUNNER
# ─────────────────────────────────────────────────────────────

def run_simulation(seed: int = 42, save_log: bool = True) -> dict:
    """
    Run the full 20-step AeroNet Lite simulation.

    Steps 1-3  : Grid init, CSP validation + repair, fleet selection
    Steps 4-6  : Delivery generation, no-fly placement, drone assignment
    Steps 7-10 : Drone movement
    Steps 11-14: Disruption injection and rerouting
    Steps 15-17: ML demand forecast, optional new delivery, movement
    Steps 18-19: Anomaly detection, emergency return
    Step  20   : Final summary

    Returns {"completed", "delayed", "failed", "unassigned"} counts.
    """
    random.seed(seed)
    log_lines: list[str] = []

    def _log(msg: str = "") -> None:
        print(msg)
        log_lines.append(msg)

    _log("=" * 62)
    _log(" AERONET LITE -- 20-STEP DELIVERY SIMULATION")
    _log("=" * 62)

    # ── Steps 1-3: Setup ─────────────────────────────────────
    _log("\n[Setup]")

    grid = Grid(10, 10, zone_limits={
        Zone.RESIDENTIAL: 10, Zone.COMMERCIAL: 10,
        Zone.HOSPITAL: 3, Zone.SCHOOL: 5, Zone.INDUSTRIAL: 8,
    })
    grid.populate_grid()

    errors = validate_layout(grid)
    if errors:
        _log(f"Step 1: Layout validation FAILED ({len(errors)} violation(s)). Repairing...")
    else:
        _log("Step 1: Layout validation passed.")

    repair_layout(grid)
    errors_after = validate_layout(grid)
    _log(
        "Step 2: Layout repaired -- all CSP constraints satisfied."
        if not errors_after
        else f"Step 2: Layout partially repaired. {len(errors_after)} violation(s) remain."
    )

    fleet = select_fleet_ga(grid, budget=15_000, seed=seed)
    drone_slots = _build_drone_slots(fleet)
    _log(
        f"Step 3: Fleet selected: "
        f"{fleet.light_count} light drone(s), {fleet.heavy_count} heavy drone(s). "
        f"Total cost: ${fleet.total_cost}."
    )

    # ── Steps 4-6: Delivery Assignment ───────────────────────
    _log("\n[Delivery Assignment]")

    deliveries = pick_multiple_delivery_waypoints(grid, n=6)
    _log(f"Step 4: {len(deliveries)} delivery job(s) generated.")

    reserved: set[tuple[int, int]] = set()
    for h, p, d in deliveries:
        reserved.update([h, p, d])
    clear_no_fly_at(grid, list(reserved))
    _sprinkle_no_fly(grid, n=10, seed=seed, avoid=reserved)
    clear_no_fly_at(grid, list(reserved))
    _log("Step 5: No-fly zones placed. Assigning drones with range constraints...")

    assignments = _assign_deliveries(grid, deliveries, drone_slots)
    ok_assignments = [a for a in assignments if a["status"] == "OK"]

    for a in assignments:
        if a["status"] == "OK":
            _log(
                f"  {a['job']} -> {a['drone']} ({a['type']}): "
                f"hub{a['hub']} -> pickup{a['pickup']} -> drop-off{a['dropoff']} "
                f"[{a['steps']} steps]"
            )
        else:
            _log(f"  {a['job']} UNASSIGNED -- no drone has sufficient range.")

    drone_states: list[DroneState] = []
    for a in ok_assignments:
        drone = next(s for s in drone_slots if s["id"] == a["drone"])
        ds = _make_drone_state(a, drone)
        if ds:
            drone_states.append(ds)

    _log(f"Step 6: {len(drone_states)} drone(s) launched. {len(assignments) - len(ok_assignments)} job(s) unassigned.")

    # ── Steps 7-10: Movement ─────────────────────────────────
    _log("\n[Movement Phase]")

    for step in range(7, 11):
        events = _advance_drones(drone_states, step)
        in_f = sum(1 for d in drone_states if d.status == "in_flight")
        done = sum(1 for d in drone_states if d.status == "completed")
        _log(f"Step {step}: Drones advance (2 cells/step).  In-flight: {in_f}  Completed: {done}")
        for e in events:
            _log(e)

    # ── Steps 11-14: Disruption & Rerouting ──────────────────
    _log("\n[Disruption & Rerouting]")

    dis_cell = _pick_disruption_cell(drone_states)
    if dis_cell:
        for line in _disrupt_and_reroute(grid, drone_states, dis_cell, step=11):
            _log(line)
    else:
        _log("Step 11: No suitable cell found on active routes -- skipping disruption.")

    for step in range(12, 15):
        events = _advance_drones(drone_states, step)
        in_f = sum(1 for d in drone_states if d.status == "in_flight")
        done = sum(1 for d in drone_states if d.status == "completed")
        _log(f"Step {step}: Movement.  In-flight: {in_f}  Completed: {done}")
        for e in events:
            _log(e)

    # ── Steps 15-17: ML Demand Forecast ──────────────────────
    _log("\n[ML Demand Forecast]")

    demand_df = generate_demand_dataset(n_samples=800, seed=seed)
    demand_res = train_demand_models(demand_df)
    rf_demand = demand_res["Random Forest"]
    _log(
        f"Step 15: Demand model trained.  "
        f"MAE={rf_demand['mae']:.2f}  RMSE={rf_demand['rmse']:.2f}"
    )
    demand_forecast_for_grid(
        grid, rf_demand["model"], hour=14, day_of_week=2, temperature=28.0
    )
    _log("Step 16: Grid demand updated from ML forecast (14:00, weekday, 28 C).")

    events = _advance_drones(drone_states, 17)
    in_f = sum(1 for d in drone_states if d.status == "in_flight")
    done = sum(1 for d in drone_states if d.status == "completed")
    _log(f"Step 17: Movement.  In-flight: {in_f}  Completed: {done}")
    for e in events:
        _log(e)

    # ── Steps 18-19: Anomaly Detection ───────────────────────
    _log("\n[Anomaly Detection]")

    tele_df = generate_telemetry_dataset(n_normal=500, n_anomalies=300, seed=seed)
    anomaly_res = train_anomaly_models(tele_df)
    rf_anomaly = anomaly_res["Random Forest"]
    _log(
        f"Step 18: Anomaly classifier trained.  "
        f"Accuracy={rf_anomaly['accuracy']:.1%}"
    )
    for line in _inject_anomaly(grid, drone_states, rf_anomaly["model"], step=18):
        _log(line)

    events = _advance_drones(drone_states, 19)
    in_f = sum(1 for d in drone_states if d.status in ("in_flight", "returning"))
    done = sum(1 for d in drone_states if d.status == "completed")
    _log(f"Step 19: Movement.  In-flight/returning: {in_f}  Completed: {done}")
    for e in events:
        _log(e)

    # ── Step 20: Final Summary ────────────────────────────────
    _log("\n[Final Summary]")

    completed  = sum(1 for d in drone_states if d.status == "completed")
    delayed    = sum(1 for d in drone_states if d.status in ("in_flight", "returning", "returned"))
    failed     = sum(1 for d in drone_states if d.status == "failed")
    unassigned = len(assignments) - len(ok_assignments)

    _log(f"Step 20: Simulation complete.")
    _log(f"  Deliveries completed : {completed}")
    _log(f"  Deliveries delayed   : {delayed}  (still in transit or forced return)")
    _log(f"  Deliveries failed    : {failed}  (no safe route)")
    _log(f"  Jobs unassigned      : {unassigned}  (range limit)")
    _log(f"  Total jobs generated : {len(assignments)}")
    _log("")
    _log(f"  {'Drone':<6} {'Job':<5} {'Status':<12} {'Position':<10} {'Progress':>8}  Anomaly")
    _log("  " + "-" * 58)
    for d in drone_states:
        anom = d.anomaly or "-"
        _log(
            f"  {d.drone_id:<6} {d.job_id:<5} {d.status.upper():<12} "
            f"{str(d.current_pos):<10} {d.progress_pct:>7}%  {anom}"
        )
    _log("=" * 62)

    if save_log:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("\n".join(log_lines), encoding="utf-8")
        print(f"\n[Log saved -> {LOG_PATH.relative_to(_ROOT)}]")

    return {
        "completed":  completed,
        "delayed":    delayed,
        "failed":     failed,
        "unassigned": unassigned,
    }
