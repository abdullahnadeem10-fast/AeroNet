# AeroNet Lite — Full Project Plan
### Autonomous Drone Delivery Simulation with Web UI

---

## Tech Stack Overview

| Layer | Technology |
|---|---|
| Simulation & AI | Python (A*, CSP, GA, ML) |
| Backend API | FastAPI (exposes simulation as REST endpoints) |
| Frontend UI | React + Tailwind CSS |
| Grid Visualization | HTML Canvas or D3.js |
| ML Models | scikit-learn |
| Communication | HTTP (fetch / axios) |

> **How it works:** Python handles all AI logic. FastAPI wraps it as an API. The React website calls that API to run the simulation step-by-step and displays the results live on screen.

---

## Project Phases

---

### Phase 1 — Project Setup & City Grid
**Days 1–2**

**Goal:** Get the foundation ready — folder structure, shared grid model, and basic zone map rendered on the website.

#### Backend Tasks
- [ ] Create project folder structure (`/src`, `/api`, `/data`, `/frontend`)
- [ ] Define the 10×10 grid in `grid_model.py` using Python dataclasses
- [ ] Each cell stores: `zone`, `density`, `is_hub`, `is_charging`, `is_medical_pickup`, `no_fly`, `demand`
- [ ] Manually define a sample city layout (hardcode for now)
- [ ] Create FastAPI app in `api/main.py`
- [ ] Add endpoint `GET /api/grid` — returns the full grid as JSON

#### Frontend Tasks
- [ ] Set up React project with Tailwind CSS
- [ ] Build `GridMap` component — renders 10×10 colored grid from API data
- [ ] Each zone type gets a distinct color (residential = green, industrial = gray, hospital = red, etc.)
- [ ] Add a legend panel showing zone color codes
- [ ] Display grid on the homepage when the page loads

#### Deliverable
Working website that loads and displays the city grid on screen.

---

### Phase 2 — CSP Layout Validation
**Days 3–4**

**Goal:** Validate the city layout against constraint rules and show pass/fail results on the website.

#### Backend Tasks
- [ ] Implement `layout_validator.py` with 4 constraint rules:
  - `R1` — Industrial cells not adjacent to schools or hospitals
  - `R2` — Every residential cell within 3 Manhattan distance of a hub
  - `R3` — Every hub within 2 cells of a charging pad
  - `R4` — At least one hospital has a medical pickup within 1 cell
- [ ] Collect all violations into a list (don't stop at first error)
- [ ] Add endpoint `POST /api/validate` — returns list of passed/failed rules with cell coordinates

#### Frontend Tasks
- [ ] Add "Validate Layout" button on the UI
- [ ] On click: call `/api/validate` and display results in a side panel
- [ ] Show each rule as a green (pass) or red (fail) badge
- [ ] On failure: highlight the offending cells on the grid map
- [ ] Show suggested fix text below each failed rule

#### Deliverable
User clicks Validate → grid highlights problem cells → panel shows all rule results.

---

### Phase 3 — Fleet Selection & A* Route Planning
**Days 5–8**

**Goal:** Let the user configure a drone fleet and plan delivery routes, displayed as animated paths on the grid.

#### Backend Tasks
- [ ] Implement `fleet_selector.py`
  - Two drone types: Light ($1000, 2 kg, 12 cells) and Heavy ($1800, 5 kg, 20 cells)
  - Brute-force or Genetic Algorithm to maximize coverage under budget
  - Fitness: `score = (0.75 × coverage%) − (0.25 × budget_used%)`
- [ ] Implement `astar_planner.py`
  - State: (row, col), Actions: up/down/left/right
  - Cost: 1.0 normal cell, 0.8 commercial corridor
  - Heuristic: Manhattan distance
  - Skip cells where `no_fly = True`
  - Returns: path coordinates + total cost
- [ ] Generate 5–10 random deliveries (hub → pickup → dropoff → hub)
- [ ] Add endpoint `POST /api/fleet` — accepts budget, returns selected fleet
- [ ] Add endpoint `POST /api/routes` — returns planned paths for all deliveries

#### Frontend Tasks
- [ ] Add "Fleet Config" panel — budget input slider, select algorithm (brute-force or GA)
- [ ] "Plan Fleet" button → calls `/api/fleet` → shows selected drone count + cost breakdown
- [ ] "Plan Routes" button → calls `/api/routes` → draws paths on the grid as colored lines
- [ ] Each drone gets a different color path
- [ ] Show delivery assignment table: Drone ID | Pickup | Dropoff | Route Cost
- [ ] Animate drone icon moving along its path (step-by-step or smooth)

#### Deliverable
User sets budget → sees fleet selected → sees all delivery routes drawn on the grid.

---

### Phase 4 — Disruption Handling & Re-routing
**Day 9–10**

**Goal:** Simulate a mid-flight disruption and show live re-routing on the website.

#### Backend Tasks
- [ ] Implement `disruption_handler.py`
  - Accept a cell coordinate to mark as no-fly
  - Detect which active drone routes pass through that cell
  - Re-run A* from the drone's current position for affected drones
  - Return: updated routes, reroute events, any failed deliveries
- [ ] Add endpoint `POST /api/disrupt` — accepts `{row, col, step}` → returns updated simulation state

#### Frontend Tasks
- [ ] Add "Trigger Disruption" button (or allow user to click a cell to mark it as no-fly)
- [ ] Clicked cell turns red/blocked on the grid instantly
- [ ] Affected drone paths redraw with new routes (in a different color/dashed style)
- [ ] Event log panel updates: "Step X: Drone D2 rerouted via A*"
- [ ] If no safe route exists: show "Delivery failed" badge on that drone

#### Deliverable
User clicks a cell → it becomes no-fly → affected drones visibly re-route on screen.

---

### Phase 5 — Machine Learning Pipeline
**Days 11–13**

**Goal:** Train demand forecasting and anomaly detection models, and display results on the website.

#### Backend Tasks
- [ ] Implement `ml_pipeline.py`

**Demand Forecasting:**
  - Load Bike Sharing Demand dataset from Kaggle
  - Features: hour, day, temperature, weather
  - Train Linear Regression or Random Forest Regressor
  - Report MAE and RMSE
  - Use predicted demand to influence delivery generation

**Anomaly Detection:**
  - Generate synthetic drone telemetry: battery_drop, speed, route_deviation, altitude_change
  - Label anomalies: Normal / Battery anomaly / Route anomaly / Sensor spike
  - Train Decision Tree or Random Forest classifier
  - Report accuracy + confusion matrix

- [ ] Add endpoint `GET /api/ml/demand` — returns forecast values per grid zone
- [ ] Add endpoint `GET /api/ml/anomalies` — returns anomaly labels per drone per step

#### Frontend Tasks
- [ ] Add "ML Insights" tab in the UI
- [ ] Demand heatmap: overlay predicted demand values on the grid (color intensity = demand level)
- [ ] Anomaly panel: table showing Drone ID | Step | Anomaly Type | Status
- [ ] Highlight anomalous drones on the grid with a warning icon
- [ ] Display model performance: MAE/RMSE card + confusion matrix table

#### Deliverable
ML tab shows demand heatmap over the grid + anomaly alerts per drone.

---

### Phase 6 — Full Simulation Runner & Final Integration
**Day 14**

**Goal:** Wire all modules together into a single 20-step simulation that runs from the website.

#### Backend Tasks
- [ ] Implement `delivery_simulator.py` — orchestrates all 5 modules in sequence
- [ ] 20-step simulation loop:
  - Steps 1–3: Validate grid, select fleet
  - Steps 4–6: Generate deliveries, compute routes
  - Steps 7–10: Move drones along paths
  - Step 11: Activate a no-fly cell
  - Steps 12–14: Re-route affected drones
  - Steps 15–17: Run demand forecast, add delivery if needed
  - Step 18: Inject or detect anomaly
  - Step 19: Reroute or return drone to hub
  - Step 20: Final summary
- [ ] Add endpoint `POST /api/simulate` — runs full 20-step sim, returns step-by-step event log

#### Frontend Tasks
- [ ] "Run Full Simulation" button on homepage
- [ ] Steps play out one by one with a short delay between each (or Next Step button)
- [ ] Grid updates live at each step (drone positions, no-fly zones, route changes)
- [ ] Event log panel streams each step's message as it happens
- [ ] Final summary card: Completed | Delayed | Failed deliveries
- [ ] Export button: download event log as `.txt`

#### Deliverable
Full working simulation running from the browser, step by step, with live grid updates.

---

## Website Pages / Screens

| Page | Purpose |
|---|---|
| `/` Home | Grid map, Run Simulation button, status panel |
| `/validate` | Layout constraint checker with highlighted violations |
| `/fleet` | Budget config, drone selection, route planner |
| `/simulate` | Step-by-step 20-step simulation runner |
| `/ml` | Demand heatmap + anomaly detection results |

---

## Folder Structure

```
aeronet_lite/
  data/
    raw/
    processed/
  src/
    grid_model.py
    layout_validator.py
    fleet_selector.py
    astar_planner.py
    disruption_handler.py
    delivery_simulator.py
    ml_pipeline.py
  api/
    main.py            ← FastAPI app
    routes/
      grid.py
      validate.py
      fleet.py
      simulate.py
      ml.py
  frontend/
    src/
      components/
        GridMap.jsx
        EventLog.jsx
        FleetPanel.jsx
        MLInsights.jsx
        AnomalyTable.jsx
      pages/
        Home.jsx
        Validate.jsx
        Fleet.jsx
        Simulate.jsx
        ML.jsx
      App.jsx
      main.jsx
    public/
    package.json
  notebooks/
    demand_forecasting.ipynb
    anomaly_classifier.ipynb
  report/
    final_report.docx
  README.md
```

---

## Day-by-Day Timeline

| Day | Focus | Deliverable |
|---|---|---|
| 1 | Grid model + FastAPI setup | `GET /api/grid` working |
| 2 | React frontend + GridMap component | City grid renders in browser |
| 3 | CSP rules R1 & R2 | Partial validation report |
| 4 | CSP rules R3 & R4 + violation highlights | Full validator with UI feedback |
| 5 | Fleet selector (brute-force) + fitness score | Fleet chosen under budget |
| 6 | GA fleet selector + API endpoint | `/api/fleet` working |
| 7 | A* path planner | Shortest path between two cells |
| 8 | Full delivery routing + route overlay on grid | All drone routes drawn |
| 9 | Disruption handler backend | `/api/disrupt` working |
| 10 | Disruption UI — click-to-block + reroute animation | Live rerouting on screen |
| 11 | Load dataset + train regression model | MAE/RMSE reported |
| 12 | Synthetic anomaly data + classifier | Confusion matrix output |
| 13 | ML endpoints + heatmap + anomaly UI | ML tab complete |
| 14 | Full 20-step simulation + integration + polish | Simulation runs end-to-end in browser |

---

## Rubric Alignment

| Rubric Area | How AeroNet Lite (with Web UI) Satisfies It |
|---|---|
| Design document | Each module, algorithm, API route, and UI screen is documented |
| Technical implementation | Python AI logic + FastAPI backend + React frontend all integrated |
| AI concept coverage | CSP, A* search, GA optimization, regression, classification |
| Viva and defense | Algorithms are explainable; website makes it easy to demo live |
| Interface and presentation | Full website with grid map, route overlay, heatmap, anomaly panel, event log |

---

## Final Submission Checklist

- [ ] Working Python simulation (`delivery_simulator.py`)
- [ ] FastAPI backend with all 5 route groups
- [ ] React website with 5 pages/screens
- [ ] 10×10 grid rendered in the browser
- [ ] CSP validator with visual violation highlights
- [ ] Fleet selection under budget with route overlay
- [ ] Live re-routing after disruption
- [ ] Demand heatmap from regression model
- [ ] Anomaly table from classifier
- [ ] 20-step simulation running step-by-step in browser
- [ ] Event log + final summary card
- [ ] Short final report