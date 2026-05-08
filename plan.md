# AeroNet Lite — Full Project Plan
### Autonomous Drone Delivery Simulation (Python)

---

## Tech Stack Overview

| Layer | Technology |
|---|---|
| Simulation & AI | Python (A*, CSP, GA, ML) |
| Grid & validation | `grid_model.py`, `layout_validator.py` |
| Visualization | Matplotlib (`visualization.py`) |
| ML Models | scikit-learn |
| Entry points | `src/main.py`, `main.ipynb`, notebooks |

> **How it works:** All logic runs in Python. Run `python src/main.py` or use Jupyter; save plots to `report/figures/` when needed.

---

## Project Phases

---

### Phase 1 — Project Setup & City Grid
**Days 1–2**

**Goal:** Folder structure, shared grid model, and zone map rendered with Matplotlib.

#### Tasks
- [ ] Project layout: `data/raw`, `data/processed`, `src/`, `notebooks/`, `report/figures` (see **Folder Structure)
- [ ] `grid_model.py`: zones (`Zone` enum), `Cell`, `Grid` with `populate_grid`, neighbors, Manhattan distance
- [ ] Cells carry: `zone`, `density`, `is_hub`, `is_charging`, `is_medical_pickup`, `no_fly`, `demand`
- [ ] `visualization.py`: `plot_zone_layout` — legend, zone colors
- [ ] `main.py` / `main.ipynb`: generate grid and show figure

#### Deliverable
Running script or notebook that displays the colored 10×10 zone layout.

---

### Phase 2 — CSP Layout Validation
**Days 3–4**

**Goal:** Validate layouts against constraint rules; print or log pass/fail; optional repair loop.

#### Tasks
- [ ] `layout_validator.py` — four rules:
  - `R1` — Industrial not adjacent to school or hospital
  - `R2` — Every residential cell within Manhattan distance 3 of a hub
  - `R3` — Every hub within distance 2 of a charging cell
  - `R4` — At least one hospital has a medical pickup within distance 1
- [ ] Collect all violations (do not stop at first error)
- [ ] `repair_layout` / auto-fix loop where appropriate
- [ ] `print_validation_report` for console/notebook output

#### Deliverable
Validation report before/after repair; layout still plottable.

---

### Phase 3 — Fleet Selection & A* Route Planning
**Days 5–8**

**Goal:** Choose a fleet under budget and plan routes; draw routes on the grid (Matplotlib).

#### Tasks
- [ ] `fleet_selector.py` — Light ($1000, 2 kg, 12 cells) vs Heavy ($1800, 5 kg, 20 cells); brute-force or GA; fitness `score = (0.75 × coverage%) − (0.25 × budget_used%)`
- [ ] `astar_planner.py` — 4-neighbor moves; cost 1.0 default, 0.8 on commercial; Manhattan heuristic; respect `no_fly`
- [ ] Generate several deliveries (hub → pickup → dropoff → hub)
- [ ] Extend `visualization.py` (or a small helper) to overlay paths

#### Deliverable
Script/notebook: fleet summary + route polylines on the zone map.

---

### Phase 4 — Disruption Handling & Re-routing
**Day 9–10**

**Goal:** Mark a cell no-fly mid-simulation and re-plan affected routes.

#### Tasks
- [ ] `disruption_handler.py` (or module under `src/`) — mark `(row, col)` no-fly; find affected routes; re-run A*; record failures
- [ ] Print or log reroute events for the demo notebook/report

#### Deliverable
Repeatable scenario: disruption → updated paths + event messages.

---

### Phase 5 — Machine Learning Pipeline
**Days 11–13**

**Goal:** Demand forecasting + anomaly detection; metrics and optional grid overlays.

#### Tasks
- [ ] `ml_pipeline.py`
- **Demand:** Bike-sharing or similar features; regression; MAE/RMSE; tie forecasts to delivery generation if useful
- **Anomaly:** Synthetic telemetry; classifier; accuracy + confusion matrix
- [ ] Plots/tables saved under `report/figures/`; integrate snippets in `notebooks/`

#### Deliverable
Notebook or script showing metrics + at least one figure for the report.

---

### Phase 6 — Full Simulation Runner & Final Integration
**Day 14**

**Goal:** One orchestrated run (e.g. 20-step narrative) in `delivery_simulator.py`.

#### Tasks
- [ ] `delivery_simulator.py` — sequence: validate → fleet → routes → movement steps → disruption → reroute → ML hooks → summary
- [ ] Step log to stdout or a `.txt` file under `report/` for submission
- [ ] Polish `main.py` or a dedicated runner notebook

#### Deliverable
End-to-end demo without a web server; documented steps for viva.

---

## Running Locally

| Entry | Purpose |
|---|---|
| `python src/main.py` | Grid demo: validate, repair, zone plot |
| `main.ipynb` | Same pipeline from the notebook |
| `notebooks/demand_forecasting.ipynb` | ML — demand |
| `notebooks/anomaly_classifier.ipynb` | ML — anomalies |

---

## Folder Structure

```
aeronet_lite/
├── README.md
├── main.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── grid_model.py
│   ├── layout_validator.py
│   ├── fleet_selector.py
│   ├── astar_planner.py
│   ├── delivery_simulator.py
│   ├── ml_pipeline.py
│   ├── visualization.py
│   └── main.py
├── notebooks/
│   ├── demand_forecasting.ipynb
│   └── anomaly_classifier.ipynb
└── report/
    ├── figures/
    └── final_report.docx
```

---

## Day-by-Day Timeline

| Day | Focus | Deliverable |
|---|---|---|
| 1 | Grid model + structure | `grid_model.py`, populated grid |
| 2 | Matplotlib zone map | `visualization.py`, `main.py` |
| 3 | CSP R1 & R2 | Partial validation in `layout_validator.py` |
| 4 | CSP R3 & R4 + repair | Full validator + report |
| 5 | Fleet selector (brute-force) | Fleet under budget |
| 6 | GA fleet variant | Tuned selector |
| 7 | A* planner | Paths on grid |
| 8 | Full routing + plot | Multi-drone overlay |
| 9 | Disruption handler | Re-plan logic |
| 10 | Disruption demo | Log + updated plot |
| 11 | Regression dataset + model | MAE/RMSE |
| 12 | Anomaly data + classifier | Confusion matrix |
| 13 | ML notebooks + figures | `report/figures/` |
| 14 | `delivery_simulator.py` + polish | Full run + summary |

---

## Rubric Alignment

| Rubric Area | How AeroNet Lite Satisfies It |
|---|---|
| Design document | Modules, algorithms, and run paths documented here |
| Technical implementation | Python-only pipeline; notebooks for ML |
| AI concept coverage | CSP, A*, GA, regression, classification |
| Viva and defense | Algorithms runnable from CLI/notebook |
| Interface and presentation | Matplotlib visuals, notebooks, written report |

---

## Final Submission Checklist

- [ ] Working Python simulation (`delivery_simulator.py` when phases complete)
- [ ] Grid + CSP validation (`grid_model.py`, `layout_validator.py`)
- [ ] Fleet + A* (`fleet_selector.py`, `astar_planner.py`)
- [ ] Disruption / reroute behavior (where required)
- [ ] ML pipeline outputs + figures
- [ ] Step-by-step or batch simulation log
- [ ] `report/final_report.docx` (and figures)
