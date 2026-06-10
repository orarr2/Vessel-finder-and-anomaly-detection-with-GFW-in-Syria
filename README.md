# Vessel Routes Finder — Anomaly Detection & Illegal Fishing Activity

An end-to-end **data-science / ML** project (in a single Jupyter notebook) that
ingests vessel tracking (AIS) data, engineers behavioural features, and flags
**anomalous vessel behaviour** that may indicate **illegal, unreported and
unregulated (IUU) fishing** or other suspicious activity such as:

- **AIS "dark" gaps** — a vessel switching off its transponder to go dark
- **Loitering / rendezvous at sea** — possible transshipment between vessels
- **Fishing inside Marine Protected Areas (MPAs)** or other restricted zones
- **Implausible position jumps** — possible AIS spoofing

The pipeline uses **[Global Fishing Watch (GFW)](https://globalfishingwatch.org/our-apis/)**
as its real-world data source, and ships with a **synthetic AIS generator** so
the whole notebook runs end-to-end *without* a token while you wait for API
access.

---

## Quick start

> **Every user runs this with their own GFW token.** No token is shipped in
> the repo. See `INSTRUCTIONS.txt` for the step-by-step walkthrough.

```bash
git clone https://github.com/orarr2/Vessel-finder-and-anomaly-detection-with-GFW-in-Syria.git
cd Vessel-finder-and-anomaly-detection-with-GFW-in-Syria
python -m venv .venv && source .venv/bin/activate     # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Get your own GFW token (free, non-commercial)

1. Sign up at <https://globalfishingwatch.org/our-apis/> and request an API token.
2. Provide the token via a git-ignored `.env` file (auto-loaded) **or** as an
   environment variable:

```bash
cp .env.example .env        # then edit in your token, OR:
export GFW_API_TOKEN="your_token_here"     # Windows PS: $env:GFW_API_TOKEN="..."
```

The `.env` file is git-ignored — your token never leaves your machine.

### Pull data and run the model (no Jupyter needed)

```bash
python fetch_syria.py                                     # default range
python fetch_syria.py --start 2024-01-01 --end 2024-12-31 --limit 1000
python analyze_syria.py                                   # runs IF + LOF + rules
```

Outputs in `./data/` (git-ignored): `syria_vessels.csv`, `syria_events.csv`,
`syria_fishing_effort.json`, and the model output `syria_vessel_risk.csv`
(per-vessel scores + risk flag).

### Or run the full notebook (EDA + interactive maps)

```bash
jupyter notebook vessel_anomaly_detection.ipynb
```

The notebook runs out-of-the-box on synthetic data; with a token set, Section 3
also pulls real Syria data.

---

## What the notebook does

| Step | Content |
|------|---------|
| 1 | Setup & dependencies (`%pip install` runs inside the notebook) |
| 2 | Connect to the GFW API (Vessels / Events / 4Wings) |
| 3 | **Live data pull for Syria** — vessels flagged `SYR`, events + fishing effort in the Syrian EEZ |
| 4 | Synthetic AIS fleet generator (normal + injected IUU behaviours) |
| 5 | EDA & route mapping (matplotlib + interactive Folium) |
| 6 | Feature engineering (speed, turning, AIS gaps, jumps, distance-to-port, zone tests) |
| 7 | Unsupervised anomaly detection — **Isolation Forest + LOF + DBSCAN** ensemble |
| 8 | Rule-based detectors for known IUU patterns |
| 9 | Visualising & interpreting flagged tracks |
| 10 | Evaluation against ground-truth labels |
| 11 | Next steps (sequence models, real geospatial layers, productionising) |

### Syria live-data section

Section 3 uses your token to pull **real Syria data**:
- **Vessels** flagged to Syria (`where flag = 'SYR'`)
- **Events** (`fishing`, `encounter`, `loitering`, `gap`) inside a GeoJSON polygon
  over the Syrian EEZ (Levantine Sea)
- **Apparent fishing effort** (4Wings monthly report) over the same area

> ⚠️ **Network requirement:** the GFW host (`globalfishingwatch.org`) must be
> reachable. On a restricted network (e.g. a sandbox with an allowlist) you'll
> get `Host not in allowlist` / HTTP 403 — run the notebook on a machine with
> normal internet, or widen the allowlist. The cells degrade gracefully: they
> print the error and the notebook continues to the synthetic demo.

> The EEZ is approximated by a bounding box; for production swap in the official
> Syria EEZ polygon (Marine Regions / GFW `public-eez-areas`).

## The Global Fishing Watch API in one table

| API | What it gives you |
|-----|-------------------|
| **4Wings** | Gridded AIS apparent fishing-effort rasters & time series |
| **Vessels** | Search vessels by name / MMSI / IMO, get identity & history |
| **Events** | `fishing`, `encounter` (rendezvous), `loitering`, `port_visit`, `gap` (AIS off) |
| **Insights** | Per-vessel risk indicators |

The **Events** API is the most directly useful for IUU work — `encounter`,
`loitering`, and `gap` events are exactly the behaviours investigators care about.

## Responsible use

AIS gaps, loitering, and zone presence are **indicators, not proof**. Treat the
output as **leads for human review**, document your assumptions, and account for
false positives (legitimate slow speed, genuine AIS outages, etc.).

---

## Project structure

```
.
├── README.md
├── INSTRUCTIONS.txt
├── requirements.txt
├── vessel_anomaly_detection.ipynb   # full EDA + modelling notebook
├── fetch_syria.py                   # GFW data puller (vessels + events + effort)
├── analyze_syria.py                 # runs IF + LOF + rules on the pulled data
├── .env.example
└── .gitignore
```

*Data attribution: Global Fishing Watch. Built as an educational reference pipeline.*
