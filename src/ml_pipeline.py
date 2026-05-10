"""Machine learning pipeline - Phase 5.
Demand forecasting (regression) and flight anomaly detection (classification).

Demand data: UCI / Kaggle Bike Sharing Dataset (hour.csv).
  Source : https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset
  File   : data/raw/bike_sharing_hour.csv  (downloaded at first run)
  Columns used: hr, weekday, temp, weathersit, cnt
  Synthetic columns added: zone_type, density, is_hub
    (these have no equivalent in the real dataset; kept for grid integration)
"""

from __future__ import annotations

from pathlib import Path
import sys
import urllib.request
import zipfile
import io

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

FIGURES_DIR = _ROOT / "report" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

_BIKE_RAW_PATH = _ROOT / "data" / "raw" / "bike_sharing_hour.csv"
_BIKE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/00275/Bike-Sharing-Dataset.zip"
)


# ─────────────────────────────────────────────────────────────
# PART 1 - DEMAND FORECASTING
# ─────────────────────────────────────────────────────────────

DEMAND_FEATURES = [
    "hour", "day_of_week", "temperature",
    "weather", "zone_type", "density", "is_hub",
]

# Maps grid Zone enum names -> integer index used as a feature
_ZONE_TO_IDX = {
    "RESIDENTIAL": 0, "COMMERCIAL": 1, "HOSPITAL": 2,
    "SCHOOL": 3, "INDUSTRIAL": 4, "OPEN_FIELD": 5,
}


def _ensure_bike_csv() -> None:
    """Download bike_sharing_hour.csv from UCI if not already present."""
    if _BIKE_RAW_PATH.exists():
        return
    _BIKE_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("  [download] Fetching Bike Sharing Dataset from UCI ML Repository ...")
    raw = urllib.request.urlopen(_BIKE_URL, timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        _BIKE_RAW_PATH.write_bytes(z.read("hour.csv"))
    print(f"  [saved] {_BIKE_RAW_PATH.relative_to(_ROOT)}")


def load_real_demand_dataset(n_samples: int = 800, seed: int = 42) -> pd.DataFrame:
    """
    Load UCI Bike Sharing Demand dataset (hour.csv) and map its columns to
    the AeroNet feature schema.

    Column mapping
    --------------
    hr          -> hour          (0-23, direct)
    weekday     -> day_of_week   (0-6, direct)
    temp        -> temperature   (normalized 0-1; denormalized to Celsius:
                                  t_C = temp*47 - 8, where min=-8 C, max=39 C)
    weathersit  -> weather       (1->0 clear, 2->1 cloudy, 3/4->2 rain)
    cnt         -> demand        (total rentals; scaled to 0-100 range)

    zone_type, density, is_hub are not in the Bike Sharing dataset; they are
    assigned synthetically so the model can still predict per-grid-cell demand.
    """
    _ensure_bike_csv()

    df_raw = pd.read_csv(_BIKE_RAW_PATH)

    # Reproducible shuffle then sample
    df_raw = df_raw.sample(frac=1, random_state=int(seed)).reset_index(drop=True)
    if n_samples < len(df_raw):
        df_raw = df_raw.iloc[:n_samples].copy()

    rng = np.random.default_rng(int(seed))
    n = len(df_raw)

    hour        = df_raw["hr"].values
    day_of_week = df_raw["weekday"].values
    # Denormalize: UCI formula is t_norm = (t - t_min)/(t_max - t_min), t_min=-8, t_max=39
    temperature = (df_raw["temp"].values * 47.0 - 8.0).round(1)
    # weathersit: 1=clear, 2=mist/cloudy, 3=light rain, 4=heavy rain -> 0/1/2
    weather     = np.where(df_raw["weathersit"] == 1, 0,
                  np.where(df_raw["weathersit"] == 2, 1, 2)).astype(int)
    # Normalize cnt to 0-100
    cnt_max     = df_raw["cnt"].max()
    demand      = (df_raw["cnt"].values / cnt_max * 100).round().astype(int)

    # Synthetic grid-only columns (uniform random, consistent with seed)
    zone_type   = rng.integers(0, 6, n)
    density     = rng.integers(0, 100, n)
    is_hub      = rng.integers(0, 2, n)

    return pd.DataFrame({
        "hour":        hour,
        "day_of_week": day_of_week,
        "temperature": temperature,
        "weather":     weather,
        "zone_type":   zone_type,
        "density":     density,
        "is_hub":      is_hub,
        "demand":      demand,
    })


def generate_demand_dataset(n_samples: int = 800, seed: int = 42) -> pd.DataFrame:
    """
    Fully synthetic fallback — used only when the real dataset is unavailable.
    Prefer load_real_demand_dataset() for viva/submission.
    """
    rng = np.random.default_rng(int(seed))

    hour    = rng.integers(0, 24, n_samples)
    day     = rng.integers(0, 7, n_samples)
    temp    = rng.uniform(10.0, 40.0, n_samples)
    weather = rng.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])
    zone    = rng.integers(0, 6, n_samples)
    density = rng.integers(0, 100, n_samples)
    is_hub  = rng.integers(0, 2, n_samples)

    base        = 20 + 0.3 * density
    hour_fx     = 10.0 * np.sin(np.pi * hour / 12.0)
    day_fx      = np.where(day < 5, 5.0, -5.0)
    temp_fx     = -0.02 * (temp - 25.0) ** 2
    weather_fx  = np.select([weather == 0, weather == 1], [5.0, 0.0], -10.0)
    zone_fx     = np.array([0, 12, -5, -5, 8, 0], dtype=float)[zone]
    hub_fx      = is_hub * 8.0
    noise       = rng.normal(0, 5, n_samples)

    demand = np.clip(
        base + hour_fx + day_fx + temp_fx + weather_fx + zone_fx + hub_fx + noise,
        0, 100,
    ).round().astype(int)

    return pd.DataFrame({
        "hour":        hour,
        "day_of_week": day,
        "temperature": temp.round(1),
        "weather":     weather,
        "zone_type":   zone,
        "density":     density,
        "is_hub":      is_hub,
        "demand":      demand,
    })


def train_demand_models(df: pd.DataFrame) -> dict:
    """
    Train Linear Regression and Random Forest Regressor on demand data.
    Returns a dict: {model_name -> {model, mae, rmse, y_test, y_pred}}.
    """
    X = df[DEMAND_FEATURES].values
    y = df["demand"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    specs = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
    }
    results: dict = {}
    for name, model in specs.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "model":  model,
            "mae":    mean_absolute_error(y_test, y_pred),
            "rmse":   root_mean_squared_error(y_test, y_pred),
            "y_test": y_test,
            "y_pred": y_pred,
        }
    return results


def print_demand_metrics(results: dict) -> None:
    print("\n--- Demand Forecasting Metrics ---")
    print(f"{'Model':<22} {'MAE':>8} {'RMSE':>8}")
    print("-" * 42)
    for name, r in results.items():
        print(f"{name:<22} {r['mae']:>8.2f} {r['rmse']:>8.2f}")


def plot_demand_results(
    results: dict,
    save_dir: Path | None = None,
    show: bool = True,
) -> None:
    """Actual vs predicted scatter + Random Forest feature importance."""
    _save = save_dir or FIGURES_DIR

    # Figure 1 - actual vs predicted
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, results.items()):
        ax.scatter(r["y_test"], r["y_pred"], alpha=0.4, s=16, color="#1d3557")
        ax.plot([0, 100], [0, 100], "r--", linewidth=1.4, label="perfect fit")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Actual demand", fontsize=10)
        ax.set_ylabel("Predicted demand", fontsize=10)
        ax.set_title(f"{name}\nMAE={r['mae']:.2f}  RMSE={r['rmse']:.2f}", fontsize=11)
        ax.legend(fontsize=8)
    fig.suptitle("Demand Forecasting: Actual vs Predicted", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, _save / "demand_actual_vs_pred.png", show)

    # Figure 2 - feature importance (Random Forest)
    rf = results.get("Random Forest")
    if rf:
        importances = rf["model"].feature_importances_
        idx = np.argsort(importances)[::-1]
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.bar(range(len(importances)), importances[idx], color="#2a9d8f", edgecolor="white")
        ax2.set_xticks(range(len(importances)))
        ax2.set_xticklabels(
            [DEMAND_FEATURES[i] for i in idx], rotation=30, ha="right", fontsize=9
        )
        ax2.set_ylabel("Importance", fontsize=10)
        ax2.set_title("Random Forest - Feature Importances (Demand)", fontsize=11)
        fig2.tight_layout()
        _save_fig(fig2, _save / "demand_feature_importance.png", show)


def demand_forecast_for_grid(
    grid,
    model,
    *,
    hour: int = 12,
    day_of_week: int = 1,
    temperature: float = 25.0,
    weather: int = 0,
) -> None:
    """
    Apply the trained model to each grid cell and update cell.demand.
    This links the ML forecast back into the simulation.
    """
    for row in grid.grid:
        for cell in row:
            zone_idx = _ZONE_TO_IDX.get(cell.zone.name, 0)
            features = np.array([[
                hour, day_of_week, temperature, weather,
                zone_idx, cell.density, int(cell.is_hub),
            ]])
            cell.demand = max(0, int(model.predict(features)[0]))


# ─────────────────────────────────────────────────────────────
# PART 2 - ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────

ANOMALY_LABELS = {
    0: "Normal",
    1: "Battery Anomaly",
    2: "Route Anomaly",
    3: "Sensor Spike",
}

TELEMETRY_FEATURES = [
    "battery_drop", "speed", "route_deviation",
    "altitude_change", "speed_change",
]


def generate_telemetry_dataset(
    n_normal: int = 500,
    n_anomalies: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Synthetic drone telemetry with four classes (document spec):
      Normal          - gradual battery drop, low deviation
      Battery Anomaly - battery_drop elevated; overlaps Normal at the tails
      Route Anomaly   - route_deviation elevated; overlaps Normal at the tails
      Sensor Spike    - altitude_change / speed_change spike; overlaps Normal

    Class means are intentionally close enough (≈2 std-dev separation) that
    distributions partially overlap, producing realistic accuracy (~88-95%)
    instead of trivial 100%.  Non-defining features carry realistic cross-class
    noise to prevent single-feature perfect splits.
    """
    rng = np.random.default_rng(int(seed))
    rows: list[dict] = []
    per_class = n_anomalies // 3

    def _row(bd, sp, rd, ac, sc, lbl):
        return {
            "battery_drop":    float(np.clip(bd, 0, None)),
            "speed":           float(np.clip(sp, 0, None)),
            "route_deviation": float(np.clip(rd, 0, None)),
            "altitude_change": float(ac),
            "speed_change":    float(sc),
            "label":           int(lbl),
        }

    # Normal: baseline wear — all features low, moderate noise
    for _ in range(n_normal):
        rows.append(_row(
            rng.normal(3.0, 1.2),   # battery_drop: gradual wear
            rng.normal(10.0, 1.5),  # speed: cruise
            rng.normal(1.0, 0.6),   # route_deviation: minor GPS drift
            rng.normal(0.0, 1.2),   # altitude_change: terrain variation
            rng.normal(0.0, 1.2),   # speed_change: wind gusts
            0,
        ))
    # Battery Anomaly: battery_drop elevated (~2 std above Normal) — tails overlap
    for _ in range(per_class):
        rows.append(_row(
            rng.normal(6.5, 1.5),   # battery_drop: elevated but overlaps Normal
            rng.normal(9.5, 1.5),   # speed: slightly reduced (power saving)
            rng.normal(1.2, 0.7),   # route_deviation: mostly normal
            rng.normal(0.0, 1.2),
            rng.normal(0.0, 1.2),
            1,
        ))
    # Route Anomaly: route_deviation elevated (~3 std above Normal) — tails overlap
    for _ in range(per_class):
        rows.append(_row(
            rng.normal(3.5, 1.2),   # battery_drop: slightly up (rerouting effort)
            rng.normal(10.5, 1.5),  # speed: slightly higher
            rng.normal(4.0, 1.5),   # route_deviation: elevated, overlaps Normal tails
            rng.normal(0.5, 1.3),
            rng.normal(1.0, 1.2),
            2,
        ))
    # Sensor Spike: altitude_change + speed_change elevated (~2 std above Normal)
    for _ in range(per_class):
        rows.append(_row(
            rng.normal(3.5, 1.3),   # battery_drop: slightly elevated
            rng.normal(10.0, 2.0),  # speed: more variable during spike
            rng.normal(1.5, 1.0),   # route_deviation: slightly elevated
            rng.normal(4.5, 2.0),   # altitude_change: spiked, wide spread → overlaps
            rng.normal(4.5, 2.0),   # speed_change: spiked, wide spread → overlaps
            3,
        ))

    df = pd.DataFrame(rows)
    df["label"] = df["label"].astype(int)
    return df.sample(frac=1, random_state=int(seed)).reset_index(drop=True)


def train_anomaly_models(df: pd.DataFrame) -> dict:
    """
    Train Decision Tree and Random Forest classifiers.
    Returns a dict: {model_name -> {model, accuracy, confusion_matrix, report, ...}}.
    """
    X = df[TELEMETRY_FEATURES].values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    specs = {
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    results: dict = {}
    for name, model in specs.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "model":            model,
            "accuracy":         accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "report":           classification_report(
                                    y_test, y_pred,
                                    target_names=list(ANOMALY_LABELS.values()),
                                ),
            "y_test": y_test,
            "y_pred": y_pred,
        }
    return results


def print_anomaly_metrics(results: dict) -> None:
    print("\n--- Anomaly Detection Metrics ---")
    for name, r in results.items():
        print(f"\n{name}  (accuracy={r['accuracy']:.1%})")
        print(r["report"])


def plot_anomaly_results(
    results: dict,
    save_dir: Path | None = None,
    show: bool = True,
) -> None:
    """Confusion matrix heatmaps for all classifiers, saved to report/figures/."""
    _save = save_dir or FIGURES_DIR
    labels = list(ANOMALY_LABELS.values())
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, r) in zip(axes, results.items()):
        cm = r["confusion_matrix"]
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        ax.set_title(f"{name}\nAccuracy: {r['accuracy']:.1%}", fontsize=11)
        thresh = cm.max() / 2
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > thresh else "black",
                )

    fig.suptitle("Anomaly Detection - Confusion Matrices", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, _save / "anomaly_confusion_matrix.png", show)


# ─────────────────────────────────────────────────────────────
# SHARED HELPERS + TOP-LEVEL RUNNER
# ─────────────────────────────────────────────────────────────

def _save_fig(fig, path: Path, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  [saved] {path.relative_to(_ROOT)}")
    try:
        __IPYTHON__  # noqa: F821
        if show:
            plt.show()
        else:
            plt.close(fig)
    except NameError:
        if show and matplotlib.is_interactive():
            plt.show()
        else:
            plt.close(fig)


def run_ml_pipeline(
    grid=None,
    seed: int = 42,
    save_figures: bool = True,
    show: bool = True,
) -> dict:
    """
    Full Phase 5 run:
      1. Load real Bike Sharing Demand dataset (UCI/Kaggle); auto-download if needed
      2. Train Linear Regression + Random Forest regressor; report MAE/RMSE
      3. Apply best demand model to update grid.demand (if grid provided)
      4. Generate synthetic telemetry dataset
      5. Train Decision Tree + Random Forest classifier; report accuracy + CM
    Returns {"demand": demand_results, "anomaly": anomaly_results}.
    """
    print("\n" + "=" * 55)
    print(" PHASE 5 - MACHINE LEARNING PIPELINE")
    print("=" * 55)

    save_dir = FIGURES_DIR if save_figures else None

    # --- Demand ---
    print("\n[1/4] Loading Bike Sharing Demand dataset (UCI) ...")
    try:
        demand_df = load_real_demand_dataset(n_samples=800, seed=seed)
        print(f"      Source : {_BIKE_RAW_PATH.relative_to(_ROOT)}")
    except Exception as exc:
        print(f"      [warn] Could not load real dataset ({exc}); using synthetic fallback.")
        demand_df = generate_demand_dataset(n_samples=800, seed=seed)
    print(f"      {len(demand_df)} samples | features: {DEMAND_FEATURES}")

    print("[2/4] Training demand regression models ...")
    demand_results = train_demand_models(demand_df)
    print_demand_metrics(demand_results)
    plot_demand_results(demand_results, save_dir=save_dir, show=show)

    if grid is not None:
        demand_forecast_for_grid(grid, demand_results["Random Forest"]["model"])
        print("\n  Grid cell demand values updated via Random Forest forecast.")

    # --- Anomaly ---
    print("\n[3/4] Generating drone telemetry dataset ...")
    tele_df = generate_telemetry_dataset(n_normal=500, n_anomalies=300, seed=seed)
    class_counts = tele_df["label"].value_counts().sort_index()
    for lbl, cnt in class_counts.items():
        print(f"      {ANOMALY_LABELS[lbl]:<18}: {cnt} samples")

    print("[4/4] Training anomaly classifiers ...")
    anomaly_results = train_anomaly_models(tele_df)
    print_anomaly_metrics(anomaly_results)
    plot_anomaly_results(anomaly_results, save_dir=save_dir, show=show)

    print("\n[Phase 5 complete]")
    return {"demand": demand_results, "anomaly": anomaly_results}
