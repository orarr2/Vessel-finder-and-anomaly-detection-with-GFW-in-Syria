"""
Run the anomaly-detection model on the REAL Syria GFW data.

Inputs (produced by fetch_syria.py):
  data/syria_vessels.csv : vessels flagged SYR (identity registry)
  data/syria_events.csv  : events inside the Syrian EEZ bbox (any flag)
                           types: fishing, encounter, loitering, gap

What this does:
  1. Aggregate events -> per-vessel feature matrix.
  2. Score vessels with Isolation Forest + Local Outlier Factor (ensemble).
  3. Apply the notebook's rule-based detectors (high loitering, encounters,
     AIS gaps, dispersed footprint).
  4. Combine model + rules into a final risk flag and print the top suspects.

Output: data/syria_vessel_risk.csv  (per-vessel scores + final flag)
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

DATA = Path("data")
EVENTS_CSV = DATA / "syria_events.csv"
VESSELS_CSV = DATA / "syria_vessels.csv"
OUT_CSV = DATA / "syria_vessel_risk.csv"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    ev = pd.read_csv(EVENTS_CSV, parse_dates=["start", "end"])
    ves = pd.read_csv(VESSELS_CSV)
    print(f"events : {len(ev):>5} rows | vessels (SYR-flagged): {len(ves):>5} rows")
    print("event-type counts:")
    print(ev["type"].value_counts().to_string())
    return ev, ves


def build_features(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.dropna(subset=["vessel_id"]).copy()
    ev["duration_h"] = (ev["end"] - ev["start"]).dt.total_seconds() / 3600.0

    # Pivot event-type counts per vessel.
    counts = (
        ev.pivot_table(index="vessel_id", columns="type",
                       values="event_id", aggfunc="count", fill_value=0)
        .rename(columns=lambda c: f"n_{c}")
    )
    for col in ("n_fishing", "n_encounter", "n_loitering", "n_gap"):
        if col not in counts.columns:
            counts[col] = 0

    # Per-vessel summary.
    agg = ev.groupby("vessel_id").agg(
        n_events=("event_id", "count"),
        vessel_name=("vessel_name", "first"),
        vessel_flag=("vessel_flag", "first"),
        ssvid=("ssvid", "first"),
        lat_mean=("lat", "mean"),
        lon_mean=("lon", "mean"),
        lat_std=("lat", "std"),
        lon_std=("lon", "std"),
        first_seen=("start", "min"),
        last_seen=("start", "max"),
        total_duration_h=("duration_h", "sum"),
        mean_duration_h=("duration_h", "mean"),
    )
    agg["lat_std"] = agg["lat_std"].fillna(0.0)
    agg["lon_std"] = agg["lon_std"].fillna(0.0)
    agg["active_days"] = (agg["last_seen"] - agg["first_seen"]).dt.total_seconds() / 86400.0
    agg["active_days"] = agg["active_days"].clip(lower=1.0)
    agg["events_per_day"] = agg["n_events"] / agg["active_days"]
    agg["footprint"] = np.sqrt(agg["lat_std"] ** 2 + agg["lon_std"] ** 2)

    feats = agg.join(counts, how="left").fillna(0)

    # Behavioural ratios — what the IUU literature looks at.
    feats["loiter_share"] = feats["n_loitering"] / feats["n_events"].clip(lower=1)
    feats["enc_share"] = feats["n_encounter"] / feats["n_events"].clip(lower=1)
    feats["gap_share"] = feats["n_gap"] / feats["n_events"].clip(lower=1)
    feats["fish_share"] = feats["n_fishing"] / feats["n_events"].clip(lower=1)
    return feats.reset_index()


MODEL_FEATURES = [
    "n_events", "n_fishing", "n_encounter", "n_loitering", "n_gap",
    "loiter_share", "enc_share", "gap_share",
    "events_per_day", "footprint",
    "mean_duration_h", "total_duration_h",
]


def score(feats: pd.DataFrame) -> pd.DataFrame:
    X = feats[MODEL_FEATURES].astype(float).values
    Xs = StandardScaler().fit_transform(X)

    iso = IsolationForest(
        n_estimators=300, contamination=0.10, random_state=42, n_jobs=-1
    ).fit(Xs)
    feats["iso_score"] = -iso.score_samples(Xs)  # bigger = more anomalous
    feats["iso_flag"] = (iso.predict(Xs) == -1).astype(int)

    n_neigh = min(20, max(5, len(feats) // 4))
    lof = LocalOutlierFactor(n_neighbors=n_neigh, contamination=0.10, novelty=False)
    feats["lof_flag"] = (lof.fit_predict(Xs) == -1).astype(int)
    feats["lof_score"] = -lof.negative_outlier_factor_  # bigger = more anomalous

    feats["model_votes"] = feats["iso_flag"] + feats["lof_flag"]

    # Rule-based: classic IUU markers.
    feats["rule_high_loiter"] = (feats["n_loitering"] >= 5).astype(int)
    feats["rule_encounter"] = (feats["n_encounter"] >= 1).astype(int)
    feats["rule_ais_gap"] = (feats["n_gap"] >= 1).astype(int)
    feats["rule_busy"] = (feats["n_events"] >= feats["n_events"].quantile(0.95)).astype(int)
    feats["rule_votes"] = (
        feats["rule_high_loiter"] + feats["rule_encounter"]
        + feats["rule_ais_gap"] + feats["rule_busy"]
    )

    feats["risk_score"] = (
        feats["iso_score"] / feats["iso_score"].max()
        + feats["lof_score"] / feats["lof_score"].max()
        + feats["rule_votes"] / 4.0
    )
    feats["high_risk"] = (
        (feats["model_votes"] >= 1) & (feats["rule_votes"] >= 1)
    ).astype(int)
    return feats


def report(feats: pd.DataFrame) -> None:
    n = len(feats)
    print()
    print("=" * 70)
    print(f"MODEL OUTPUT  ({n} vessels with at least one event in Syrian EEZ)")
    print("=" * 70)
    iso_n = int(feats["iso_flag"].sum())
    lof_n = int(feats["lof_flag"].sum())
    both = int(((feats["iso_flag"] == 1) & (feats["lof_flag"] == 1)).sum())
    hi = int(feats["high_risk"].sum())
    print(f"  Isolation Forest flagged : {iso_n}")
    print(f"  LOF              flagged : {lof_n}")
    print(f"  Both models agree        : {both}")
    print(f"  High-risk (model+rules)  : {hi}")
    print()
    print("Flag distribution in suspect pool (top 10):")
    print(feats[feats["model_votes"] >= 1]["vessel_flag"]
          .value_counts().head(10).to_string())

    cols = [
        "vessel_name", "vessel_flag", "ssvid", "n_events",
        "n_fishing", "n_encounter", "n_loitering", "n_gap",
        "iso_flag", "lof_flag", "rule_votes", "risk_score",
    ]
    top = feats.sort_values("risk_score", ascending=False).head(15)[cols]
    print()
    print("=" * 70)
    print("TOP 15 SUSPICIOUS VESSELS")
    print("=" * 70)
    with pd.option_context("display.max_rows", None, "display.max_columns", None,
                           "display.width", 200, "display.precision", 3):
        print(top.to_string(index=False))


def main() -> None:
    ev, ves = load()
    feats = build_features(ev)
    feats = score(feats)
    feats.to_csv(OUT_CSV, index=False)
    print(f"\nWrote per-vessel risk -> {OUT_CSV}")
    report(feats)


if __name__ == "__main__":
    main()
