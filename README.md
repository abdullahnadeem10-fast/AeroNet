# AeroNet Lite — Autonomous Drone Delivery Simulation

> An AI-powered urban drone delivery system featuring pathfinding, constraint satisfaction, genetic optimization, real-time disruption handling, and ML-based demand forecasting with anomaly detection — all visualized through an interactive Streamlit dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [AI & ML Techniques](#ai--ml-techniques)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Running the Project](#running-the-project)
- [Interactive Dashboard](#interactive-dashboard)
- [Simulation Phases](#simulation-phases)
- [Sample Output](#sample-output)

---

## Overview

AeroNet Lite simulates a fleet of autonomous delivery drones operating over a 10×10 urban grid. The system handles every stage of the delivery pipeline — from validating city layout constraints and selecting the optimal drone fleet, to planning collision-free routes, recovering from mid-flight disruptions, forecasting demand, and detecting onboard anomalies.

This project was built as an integrated AI systems showcase, combining classical AI algorithms with modern machine learning in a single coherent simulation.

---

## Features

| Feature | Description |
|---|---|
| **Urban Grid Model** | 10×10 grid with 6 zone types, hub/charging/medical infrastructure |
| **CSP Layout Validation** | Enforces 4 hard spatial constraints with iterative repair |
| **Genetic Algorithm Fleet Selection** | Optimizes drone count and type under a fixed budget |
| **A\* Pathfinding** | Finds optimal delivery routes respecting no-fly zones |
| **Real-time Disruption Handling** | Re-routes affected drones mid-flight using A\* replanning |
| **Demand Forecasting** | Random Forest regression trained on synthetic telemetry |
| **Anomaly Detection** | Random Forest classifier detects battery, route, and sensor faults |
| **20-Step Simulation** | Full orchestrated run with per-step state tracking and final report |
| **Streamlit Dashboard** | Interactive UI with 6 tabs for visualization and control |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        AeroNet Lite                         │
├───────────────┬─────────────────────────────────────────────┤
│  CLI Runner   │           Streamlit Dashboard               │
│  src/main.py  │           streamlit_app.py                  │
└───────┬───────┴───────────────────────┬─────────────────────┘
        │                               │
        ▼                               ▼
┌───────────────────────────────────────────────────────────┐
│                    Core Simulation Engine                  │
├──────────────┬─────────────────────┬──────────────────────┤
│  grid_model  │  layout_validator   │   fleet_selector     │
│  (env setup) │  (CSP + repair)     │   (Genetic Algo)     │
├──────────────┼─────────────────────┼──────────────────────┤
│ astar_planner│ disruption_handler  │   ml_pipeline        │
│  (A* routing)│ (mid-flight replan) │  (RF + DT models)    │
├──────────────┴─────────────────────┴──────────────────────┤
│              delivery_simulator (20-step orchestrator)     │
├───────────────────────────────────────────────────────────┤
│                    visualization (Matplotlib)              │
└───────────────────────────────────────────────────────────┘
```

---

## AI & ML Techniques

### Classical AI

| Algorithm | Module | Purpose |
|---|---|---|
| **A\* Search** | `astar_planner.py` | Optimal drone routing with Manhattan heuristic and zone-weighted costs |
| **Constraint Satisfaction (CSP)** | `layout_validator.py` | Validates and repairs city grid against 4 hard spatial rules |
| **Genetic Algorithm** | `fleet_selector.py` | Evolves optimal fleet composition (light vs. heavy drones) under budget constraints |

### Machine Learning

| Model | Type | Module | Purpose |
|---|---|---|---|
| **Random Forest** | Regression | `ml_pipeline.py` | Forecasts per-cell delivery demand (~MAE 7–8) |
| **Linear Regression** | Regression | `ml_pipeline.py` | Baseline demand model for comparison |
| **Random Forest** | Classification | `ml_pipeline.py` | Detects drone anomalies (battery, route, sensor) |
| **Decision Tree** | Classification | `ml_pipeline.py` | Interpretable anomaly detection baseline |

### Disruption System

Real-time replanning: when a no-fly zone appears mid-flight, all affected drones immediately re-run A\* from their current position. If the new path exceeds the drone's remaining range, the delivery is marked as **failed**.

---

## Project Structure

```
AeroNet/
├── src/
│   ├── grid_model.py          # Grid, zones, cells, infrastructure flags
│   ├── layout_validator.py    # CSP constraint checking and repair
│   ├── fleet_selector.py      # Genetic algorithm for fleet optimization
│   ├── astar_planner.py       # A* pathfinding + delivery generation
│   ├── disruption_handler.py  # Mid-flight rerouting logic
│   ├── ml_pipeline.py         # Demand forecasting + anomaly detection
│   ├── delivery_simulator.py  # 20-step full simulation orchestrator
│   ├── visualization.py       # Matplotlib plots (grid, routes, heatmaps)
│   └── main.py                # CLI entry point — runs the full demo
├── report/
│   ├── simulation_log.txt     # Output from the last full simulation run
│   └── figures/               # Generated plots (confusion matrix, demand, etc.)
├── streamlit_app.py           # Interactive web dashboard (6 tabs)
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/Salar-khan-bits/AeroNet.git
cd AeroNet
pip install -r requirements.txt
```

**requirements.txt**
```
streamlit>=1.35
matplotlib>=3.8
numpy>=1.26
pandas>=2.2
scikit-learn>=1.4
```

---

## Running the Project

### Option 1 — Interactive Dashboard (Recommended)

```bash
streamlit run streamlit_app.py
```

Opens a browser UI with full control over all simulation parameters.

### Option 2 — CLI Full Demo

```bash
python src/main.py
```

Runs the complete pipeline end-to-end: grid setup → CSP repair → GA fleet selection → A\* routing → disruption handling → ML training → 20-step simulation. Plots are displayed inline and logs are saved to `report/simulation_log.txt`.

---

## Interactive Dashboard

The Streamlit dashboard is organized into 6 tabs:

| Tab | What You Can Do |
|---|---|
| **Zone Map** | View the 10×10 grid, zone distribution, and CSP validation status |
| **Delivery Routes** | Select a delivery job and visualize the A\* path with cost metrics |
| **Demand Heatmap** | Adjust hour, day, temperature, and weather to see live demand predictions |
| **Disruption** | Inject a mid-flight no-fly zone and watch the drone reroute |
| **ML Pipeline** | View training metrics, feature importance, and confusion matrix |
| **Simulation Log** | Run the 20-step simulation and read the full step-by-step log |

---

**Dashboard preview**

- **Place image:** Save your screenshot to `report/figures/dashboard.png` in the repository.
- **Markdown (simple):**

  `![Dashboard screenshot](report/figures/dashboard.png)`

- **HTML (sized):**

  `<img src="report/figures/dashboard.png" width="800" alt="Dashboard screenshot" />`

---

## Simulation Phases

The 20-step orchestrated simulation (`delivery_simulator.py`) runs through the following lifecycle:

```
Phase 1 — Setup       [Steps 1–3]   Grid → CSP Validate & Repair → GA Fleet Selection
Phase 2 — Assignments [Steps 4–6]   Generate Deliveries → Place No-Fly Zones → Assign Drones
Phase 3 — Movement    [Steps 7–10]  Advance drones 2 cells/step, track status
Phase 4 — Disruption  [Steps 11–14] Inject mid-route no-fly, A* replan, mark failures
Phase 5 — ML Demand   [Steps 15–17] Train RF demand model, update grid cell demands
Phase 6 — Anomaly     [Steps 18–19] Train anomaly classifier, inject fault, force hub return
Phase 7 — Report      [Step 20]     Completed / Delayed / Failed / Unassigned summary
```

### Grid Zone Types

| Zone | Color | Role |
|---|---|---|
| Residential | Yellow | Primary delivery destinations |
| Commercial | Blue | Air corridor zones (lower routing cost) |
| Hospital | Red | Medical pickup points |
| School | Green | Sensitive zones (no adjacent industrial) |
| Industrial | Gray | Restricted from sensitive neighbors |
| Open Field | Light Green | Buffer / unrestricted airspace |

### CSP Constraints

| Rule | Description |
|---|---|
| R1 | Industrial zones cannot be adjacent to School or Hospital |
| R2 | Every Residential cell must be within 3 cells of a hub |
| R3 | Every hub must be within 2 cells of a charging station |
| R4 | At least one Hospital must have a medical pickup within 1 cell |

### Drone Types

| Type | Cost | Payload | Range |
|---|---|---|---|
| Light | $1,000 | 2 kg | 12 cells |
| Heavy | $1,800 | 5 kg | 20 cells |

The GA optimizes `fitness = 0.75 × coverage% − 0.25 × budget_used%` over 40 generations with a population of 24.

---

## Sample Output

```
============================================================
AERONET LITE — FULL SIMULATION REPORT
============================================================
Total deliveries : 6
  Completed      : 2
  Delayed        : 2   (anomaly return / in-transit)
  Failed         : 0
  Unassigned     : 2   (range limit exceeded)

Fleet: 0 light + 8 heavy drones — $14,400 / $20,000 budget

Demand model : Random Forest  MAE=7.49  RMSE=9.61
Anomaly model: Random Forest  Accuracy=100%

Step 11 — Disruption at (2,3): D2 rerouted (+3 cells), D3 rerouted (+5 cells)
Step 18 — Battery anomaly detected on D1 → returning to hub
============================================================
```

---

## Built With

- **Python 3.10**
- **scikit-learn** — ML models
- **NumPy / Pandas** — Data processing
- **Matplotlib** — Static visualizations
- **Streamlit** — Interactive dashboard

---

*Semester 6 AI Course Project — Autonomous Systems & Intelligent Planning*
