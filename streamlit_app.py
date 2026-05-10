"""AeroNet Lite - Streamlit Dashboard."""

import sys
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")  # non-interactive backend required for Streamlit
import matplotlib.pyplot as plt
import pandas as pd

import streamlit as st

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.grid_model import Grid, Zone
from src.layout_validator import repair_layout, validate_layout
from src.fleet_selector import select_fleet_ga, LIGHT_RANGE_CELLS, HEAVY_RANGE_CELLS
from src.astar_planner import (
    pick_multiple_delivery_waypoints,
    plan_delivery_segments,
    clear_no_fly_at,
    astar,
)
from src.visualization import (
    plot_zone_layout,
    plot_delivery_routes,
    plot_demand_heatmap,
    plot_disruption_reroute,
)
from src.ml_pipeline import (
    load_real_demand_dataset,
    generate_demand_dataset,
    train_demand_models,
    demand_forecast_for_grid,
    generate_telemetry_dataset,
    train_anomaly_models,
    plot_demand_results,
    plot_anomaly_results,
    ANOMALY_LABELS,
    DEMAND_FEATURES,
    FIGURES_DIR,
    _BIKE_RAW_PATH,
)
from src.disruption_handler import make_active_delivery, activate_disruption

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AeroNet Lite",
    page_icon="drone",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AeroNet Lite - Autonomous Drone Delivery Dashboard")

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

st.sidebar.header("Simulation Controls")
seed = st.sidebar.slider("Random seed", 0, 99, 42)
budget = st.sidebar.select_slider(
    "Fleet budget ($)",
    options=[5_000, 8_000, 10_000, 12_000, 15_000, 20_000, 25_000],
    value=15_000,
)
n_deliveries = st.sidebar.slider("Number of deliveries", 2, 10, 8)

if st.sidebar.button("Run / Refresh Simulation", type="primary"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Change any control and click **Run / Refresh** to update.")

# ─────────────────────────────────────────────────────────────
# HELPERS (inlined to avoid circular imports with main.py)
# ─────────────────────────────────────────────────────────────

def _build_slots(fleet) -> list[dict]:
    slots = []
    for i in range(fleet.light_count):
        slots.append({"id": f"D{i + 1}", "type": "Light", "range": LIGHT_RANGE_CELLS})
    for j in range(fleet.heavy_count):
        slots.append({"id": f"D{fleet.light_count + j + 1}", "type": "Heavy", "range": HEAVY_RANGE_CELLS})
    return slots


def _assign(grid, deliveries, slots) -> list[dict]:
    results = []
    n = len(slots)
    for i, (hub, pickup, dropoff) in enumerate(deliveries):
        assigned = False
        for offset in range(n):
            drone = slots[(i + offset) % n]
            segs, err = plan_delivery_segments(
                grid, hub, pickup, dropoff, max_range=drone["range"]
            )
            if not err:
                results.append({
                    "job": f"J{i + 1}", "drone": drone["id"], "type": drone["type"],
                    "hub": hub, "pickup": pickup, "dropoff": dropoff,
                    "steps": sum(len(r.path) - 1 for _, r in segs if r.path),
                    "cost": sum(r.total_cost for _, r in segs),
                    "status": "OK", "segments": segs,
                })
                assigned = True
                break
        if not assigned:
            results.append({
                "job": f"J{i + 1}", "drone": "--", "type": "--",
                "hub": hub, "pickup": pickup, "dropoff": dropoff,
                "steps": 0, "cost": 0.0, "status": "FAILED", "segments": [],
            })
    return results


def _sprinkle_no_fly(grid, n=12, seed=1, avoid=None):
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


def _show(ax):
    """Display a matplotlib axes in Streamlit, then close the figure."""
    fig = ax.figure
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# SIMULATION STATE (cached in session_state)
# ─────────────────────────────────────────────────────────────

if "ready" not in st.session_state:
    random.seed(seed)

    progress = st.progress(0, text="Building grid...")
    grid = Grid(10, 10, zone_limits={
        Zone.RESIDENTIAL: 10, Zone.COMMERCIAL: 10,
        Zone.HOSPITAL: 3, Zone.SCHOOL: 5, Zone.INDUSTRIAL: 8,
    })
    grid.populate_grid()
    repair_layout(grid)
    csp_errors = validate_layout(grid)

    progress.progress(15, text="Selecting fleet (GA)...")
    fleet = select_fleet_ga(grid, budget=int(budget), seed=seed)
    slots = _build_slots(fleet)

    progress.progress(30, text="Planning deliveries...")
    deliveries = pick_multiple_delivery_waypoints(grid, n=n_deliveries)
    reserved: set = set()
    for h, p, d in deliveries:
        reserved.update([h, p, d])
    clear_no_fly_at(grid, list(reserved))
    _sprinkle_no_fly(grid, n=12, seed=seed, avoid=reserved)
    clear_no_fly_at(grid, list(reserved))
    assignments = _assign(grid, deliveries, slots)

    progress.progress(55, text="Training demand model...")
    try:
        demand_df = load_real_demand_dataset(n_samples=800, seed=seed)
    except Exception:
        demand_df = generate_demand_dataset(n_samples=800, seed=seed)
    demand_results = train_demand_models(demand_df)
    demand_forecast_for_grid(grid, demand_results["Random Forest"]["model"])
    plot_demand_results(demand_results, show=False)

    progress.progress(75, text="Training anomaly classifier...")
    tele_df = generate_telemetry_dataset(n_normal=500, n_anomalies=300, seed=seed)
    anomaly_results = train_anomaly_models(tele_df)
    plot_anomaly_results(anomaly_results, show=False)

    progress.progress(100, text="Done.")
    progress.empty()

    st.session_state.update({
        "ready": True,
        "grid": grid,
        "fleet": fleet,
        "slots": slots,
        "deliveries": deliveries,
        "assignments": assignments,
        "csp_errors": csp_errors,
        "demand_results": demand_results,
        "anomaly_results": anomaly_results,
    })

grid         = st.session_state["grid"]
fleet        = st.session_state["fleet"]
slots        = st.session_state["slots"]
deliveries   = st.session_state["deliveries"]
assignments  = st.session_state["assignments"]
csp_errors   = st.session_state["csp_errors"]
demand_results  = st.session_state["demand_results"]
anomaly_results = st.session_state["anomaly_results"]

ok_jobs  = [a for a in assignments if a["status"] == "OK"]
bad_jobs = [a for a in assignments if a["status"] != "OK"]

# ─────────────────────────────────────────────────────────────
# TOP METRIC ROW
# ─────────────────────────────────────────────────────────────

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Light drones", fleet.light_count)
m2.metric("Heavy drones", fleet.heavy_count)
m3.metric("Fleet cost", f"${fleet.total_cost:,}")
m4.metric("Jobs assigned", len(ok_jobs))
m5.metric("CSP violations", len(csp_errors), delta="after repair" if not csp_errors else None)

st.divider()

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────

tab_zone, tab_route, tab_demand, tab_disrupt, tab_ml, tab_sim = st.tabs([
    "Zone Map",
    "Delivery Routes",
    "Demand Heatmap",
    "Disruption",
    "ML Pipeline",
    "Simulation Log",
])

# ── Tab 1: Zone Map ──────────────────────────────────────────
with tab_zone:
    st.subheader("Grid Zone Layout")
    st.caption("Hatched cells (//// ) are no-fly zones. Colored by zone type.")
    col_map, col_info = st.columns([2, 1])
    with col_map:
        ax = plot_zone_layout(
            grid, show=False, mark_no_fly=True,
            title="Zone map  (//// = no-fly)",
        )
        _show(ax)
    with col_info:
        st.markdown("**Zone counts**")
        from collections import Counter
        zone_counts = Counter(cell.zone.value for row in grid.grid for cell in row)
        for zone, count in sorted(zone_counts.items()):
            st.metric(zone, count)
        st.markdown("---")
        st.markdown(f"**Grid size:** {grid.rows} x {grid.cols}")
        no_fly_count = sum(1 for row in grid.grid for cell in row if cell.no_fly)
        hub_count = sum(1 for row in grid.grid for cell in row if cell.is_hub)
        st.metric("No-fly cells", no_fly_count)
        st.metric("Hub cells", hub_count)
        if csp_errors:
            st.warning(f"{len(csp_errors)} CSP violation(s) remain after repair")
        else:
            st.success("All CSP constraints satisfied")

# ── Tab 2: Delivery Routes ───────────────────────────────────
with tab_route:
    st.subheader("A* Delivery Route Visualizer")

    if not ok_jobs:
        st.warning("No successfully assigned deliveries. Lower budget or reduce deliveries.")
    else:
        job_labels = [
            f"{a['job']} - {a['drone']} ({a['type']}, {a['steps']} steps)"
            for a in ok_jobs
        ]
        selected_label = st.selectbox("Select delivery job", job_labels)
        sel = ok_jobs[job_labels.index(selected_label)]
        segs = [(name, res.path or []) for name, res in sel["segments"]]

        ax = plot_delivery_routes(
            grid, segs,
            hub=sel["hub"], pickup=sel["pickup"], dropoff=sel["dropoff"],
            show=False,
            title=f"{sel['job']} | {sel['drone']} ({sel['type']}) | "
                  f"hub{sel['hub']} -> pickup{sel['pickup']} -> drop-off{sel['dropoff']}",
        )
        _show(ax)

        c1, c2, c3 = st.columns(3)
        c1.metric("Steps", sel["steps"])
        c2.metric("Route cost", f"{sel['cost']:.1f}")
        c3.metric("Drone type", sel["type"])

    if bad_jobs:
        st.markdown("---")
        st.markdown("**Failed assignments** (no drone had sufficient range)")
        failed_df = pd.DataFrame([
            {"Job": a["job"], "Hub": str(a["hub"]), "Pickup": str(a["pickup"]), "Drop-off": str(a["dropoff"])}
            for a in bad_jobs
        ])
        st.dataframe(failed_df, width="stretch", hide_index=True)

# ── Tab 3: Demand Heatmap ────────────────────────────────────
with tab_demand:
    st.subheader("ML-Forecast Delivery Demand Heatmap")
    st.caption("Demand values per cell predicted by the Random Forest model and applied to the grid.")

    col_heat, col_ctrl = st.columns([2, 1])
    with col_ctrl:
        hour = st.slider("Hour of day", 0, 23, 12)
        day = st.selectbox("Day of week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        day_idx = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].index(day)
        temp = st.slider("Temperature (C)", 10, 40, 25)
        weather = st.radio("Weather", ["Clear", "Cloudy", "Rain"], horizontal=True)
        weather_idx = ["Clear", "Cloudy", "Rain"].index(weather)

        if st.button("Update forecast"):
            demand_forecast_for_grid(
                grid,
                demand_results["Random Forest"]["model"],
                hour=hour, day_of_week=day_idx,
                temperature=float(temp), weather=weather_idx,
            )

    with col_heat:
        ax = plot_demand_heatmap(
            grid, show=False,
            title=f"Demand forecast  ({hour:02d}:00, {day}, {temp}C, {weather})",
        )
        _show(ax)

    demands = [cell.demand for row in grid.grid for cell in row]
    d1, d2, d3 = st.columns(3)
    d1.metric("Mean demand", f"{sum(demands)/len(demands):.1f}")
    d2.metric("Min demand", min(demands))
    d3.metric("Max demand", max(demands))

# ── Tab 4: Disruption ────────────────────────────────────────
with tab_disrupt:
    st.subheader("Disruption Handling & A* Re-routing")
    st.caption(
        "A no-fly zone is injected mid-flight. "
        "The affected drone replans its route using A* to avoid the blocked cell."
    )

    if not ok_jobs:
        st.warning("No successful deliveries available for disruption demo.")
    else:
        job_labels_d = [
            f"{a['job']} - {a['drone']} ({a['type']})" for a in ok_jobs
        ]
        sel_d_label = st.selectbox("Select delivery for disruption demo", job_labels_d)
        sel_d = ok_jobs[job_labels_d.index(sel_d_label)]

        drone_range = LIGHT_RANGE_CELLS if sel_d["type"] == "Light" else HEAVY_RANGE_CELLS
        active = make_active_delivery(sel_d, drone_range, advance_to_pickup=True)

        # Original remaining segments (from pickup onward)
        orig_segs = [(name, path) for name, path in active.segments]

        # Find a disruption cell on the remaining route
        all_cells_on_route: list[tuple[int, int]] = []
        for _, path in orig_segs:
            all_cells_on_route.extend(path)
        protected = {active.waypoints[0], active.waypoints[1], active.waypoints[2]}
        disruption_candidates = [
            c for c in all_cells_on_route
            if c not in protected and not grid.grid[c[0]][c[1]].no_fly
        ]

        if not disruption_candidates:
            st.info("No disruption candidate found on this route (all cells are waypoints or already no-fly).")
        else:
            dis_cell = disruption_candidates[len(disruption_candidates) // 2]
            st.info(f"Disruption cell: {dis_cell}  (injected as no-fly mid-flight)")

            if st.button("Trigger Disruption & Reroute"):
                log_lines = activate_disruption(
                    grid, dis_cell, [active], step=11
                )
                rerouted_segs = [(name, path) for name, path in active.segments]

                ax = plot_disruption_reroute(
                    grid,
                    orig_segs, rerouted_segs,
                    hub=sel_d["hub"],
                    pickup=sel_d["pickup"],
                    dropoff=sel_d["dropoff"],
                    disrupted_cell=dis_cell,
                    show=False,
                    title=f"Disruption at {dis_cell} - {sel_d['job']} rerouted",
                )
                _show(ax)

                st.markdown("**Event log**")
                for line in log_lines:
                    st.text(line)

                # Undo the no-fly so it doesn't break other tabs
                grid.grid[dis_cell[0]][dis_cell[1]].no_fly = False
            else:
                st.markdown(
                    "Click **Trigger Disruption & Reroute** to inject the no-fly zone "
                    "and see the drone replan its path."
                )

# ── Tab 5: ML Pipeline ───────────────────────────────────────
with tab_ml:
    st.subheader("Machine Learning Pipeline")
    ml_demand, ml_anomaly = st.tabs(["Demand Forecasting", "Anomaly Detection"])

    with ml_demand:
        st.markdown("#### Model Metrics")
        cols = st.columns(len(demand_results))
        for col, (name, r) in zip(cols, demand_results.items()):
            col.metric(name, f"MAE {r['mae']:.2f}", f"RMSE {r['rmse']:.2f}")

        st.markdown("#### Actual vs Predicted Demand")
        img1 = FIGURES_DIR / "demand_actual_vs_pred.png"
        if img1.exists():
            st.image(str(img1), width="stretch")

        st.markdown("#### Feature Importances (Random Forest)")
        img2 = FIGURES_DIR / "demand_feature_importance.png"
        if img2.exists():
            st.image(str(img2), width="stretch")

        st.markdown("#### Dataset info")
        source = "Bike Sharing Demand (UCI/Kaggle)" if _BIKE_RAW_PATH.exists() else "synthetic fallback"
        st.caption(
            f"800 samples from {source} | features: {', '.join(DEMAND_FEATURES)} | "
            "target: demand (0-100)"
        )

    with ml_anomaly:
        st.markdown("#### Classifier Accuracy")
        cols = st.columns(len(anomaly_results))
        for col, (name, r) in zip(cols, anomaly_results.items()):
            col.metric(name, f"{r['accuracy']:.1%}")

        st.markdown("#### Confusion Matrices")
        img3 = FIGURES_DIR / "anomaly_confusion_matrix.png"
        if img3.exists():
            st.image(str(img3), width="stretch")

        st.markdown("#### Anomaly Classes")
        st.table(pd.DataFrame([
            {"Class": v, "Rule": r}
            for (_, v), r in zip(
                sorted(ANOMALY_LABELS.items()),
                [
                    "Gradual battery drop (~3), low deviation — baseline",
                    "battery_drop elevated (~6.5 mean, overlaps Normal tails)",
                    "route_deviation elevated (~4.0 mean, overlaps Normal tails)",
                    "altitude_change + speed_change spike (~4.5 mean, wide spread)",
                ],
            )
        ]))

        st.markdown("#### Classification report (Random Forest)")
        st.text(anomaly_results["Random Forest"]["report"])

# ── Tab 6: Simulation Log ────────────────────────────────────
with tab_sim:
    st.subheader("20-Step Simulation Log")

    log_path = _ROOT / "report" / "simulation_log.txt"

    col_run, col_info = st.columns([1, 3])
    with col_run:
        if st.button("Run 20-step simulation", type="primary"):
            with st.spinner("Running full 20-step simulation..."):
                from src.delivery_simulator import run_simulation
                summary = run_simulation(seed=seed, save_log=True)
            st.success("Simulation complete.")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Completed", summary["completed"])
            s2.metric("Delayed", summary["delayed"])
            s3.metric("Failed", summary["failed"])
            s4.metric("Unassigned", summary["unassigned"])
            st.rerun()

    with col_info:
        if log_path.exists():
            log_text = log_path.read_text(encoding="utf-8")
            st.text_area("Log output", log_text, height=500, label_visibility="collapsed")
        else:
            st.info("No simulation log found. Click 'Run 20-step simulation' to generate one.")

    # Delivery assignment summary table
    st.markdown("---")
    st.markdown("#### Delivery Assignment Table")
    rows = []
    for a in assignments:
        rows.append({
            "Job":      a["job"],
            "Drone":    a["drone"],
            "Type":     a["type"],
            "Hub":      str(a["hub"]),
            "Pickup":   str(a["pickup"]),
            "Drop-off": str(a["dropoff"]),
            "Steps":    str(a["steps"]) if a["status"] == "OK" else "--",
            "Cost":     f"{a['cost']:.1f}" if a["status"] == "OK" else "--",
            "Status":   a["status"],
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
