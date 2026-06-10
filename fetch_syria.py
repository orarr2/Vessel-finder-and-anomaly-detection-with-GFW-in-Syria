
"""
Fetch Global Fishing Watch (GFW) data for Syria and dump it to disk.

Pulls, for Syrian waters:
  1. Vessels flagged to Syria (flag = 'SYR')               -> data/syria_vessels.csv
  2. Events in the Syrian EEZ (fishing/encounter/loitering/gap)
                                                              -> data/syria_events.csv
  3. Apparent fishing effort (4Wings monthly report)          -> data/syria_fishing_effort.json

Token resolution (never hard-code it):
    1. a .env file in the project root:   GFW_API_TOKEN=your_token
    2. an exported environment variable:  export GFW_API_TOKEN=your_token

Usage:
    pip install -r requirements.txt
    export GFW_API_TOKEN=...        # or put it in .env
    python fetch_syria.py
    python fetch_syria.py --start 2023-01-01 --end 2023-12-31 --limit 200

Requires network access to gateway.api.globalfishingwatch.org (open internet).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

GFW_BASE = "https://gateway.api.globalfishingwatch.org/v3"
OUT_DIR = Path("data")

SYRIA_FLAG = "SYR"

SYR_BBOX = {"lat_min": 34.55, "lat_max": 35.95, "lon_min": 33.50, "lon_max": 36.05}
SYR_POLYGON = {
    "type": "Polygon",
    "coordinates": [[
        [SYR_BBOX["lon_min"], SYR_BBOX["lat_min"]],
        [SYR_BBOX["lon_max"], SYR_BBOX["lat_min"]],
        [SYR_BBOX["lon_max"], SYR_BBOX["lat_max"]],
        [SYR_BBOX["lon_min"], SYR_BBOX["lat_max"]],
        [SYR_BBOX["lon_min"], SYR_BBOX["lat_min"]],
    ]],
}

EVENT_DATASETS = {
    "fishing":   "public-global-fishing-events:latest",
    "encounter": "public-global-encounters-events:latest",
    "loitering": "public-global-loitering-events:latest",
    "gap":       "public-global-gaps-events:latest",
}

def get_token() -> str:
    token = os.environ.get("GFW_API_TOKEN", "").strip()
    if not token:
        sys.exit(
            "ERROR: GFW_API_TOKEN not found.\n"
            "  Put it in a .env file (GFW_API_TOKEN=...) or run:\n"
            "    export GFW_API_TOKEN=your_token"
        )
    return token

def make_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}"})
    return s

def _check(resp: requests.Response) -> dict:
    if resp.status_code == 403 and "allowlist" in resp.text.lower():
        sys.exit(
            "ERROR: 403 'Host not in allowlist'. The network you're on is blocking\n"
            "globalfishingwatch.org. Run this on a machine with open internet."
        )
    resp.raise_for_status()
    return resp.json()

VESSEL_PAGE = 50  # GFW v3 vessels/search caps limit at 50

def fetch_vessels(session: requests.Session, flag: str = SYRIA_FLAG, limit: int = 200) -> list[dict]:
    rows = []
    since = None
    fetched = 0
    while fetched < limit:
        params = {
            "where": f"flag = '{flag}'",
            "datasets[0]": "public-global-vessel-identity:latest",
            "limit": min(VESSEL_PAGE, limit - fetched),
        }
        if since:
            params["since"] = since
        data = _check(session.get(f"{GFW_BASE}/vessels/search", params=params, timeout=90))
        entries = data.get("entries", []) or []
        if not entries:
            break
        for entry in entries:
            sri = (entry.get("selfReportedInfo") or [{}])
            reg = (entry.get("registryInfo") or [{}])
            src = sri[0] if sri else {}
            rg = reg[0] if reg else {}
            rows.append({
                "vessel_id": src.get("id") or rg.get("id") or entry.get("id"),
                "shipname": src.get("shipname") or rg.get("shipname"),
                "flag": src.get("flag") or rg.get("flag"),
                "ssvid_mmsi": src.get("ssvid") or rg.get("ssvid"),
                "imo": src.get("imo") or rg.get("imo"),
                "callsign": src.get("callsign") or rg.get("callsign"),
                "geartype": src.get("geartypes") or rg.get("geartypes"),
                "from": src.get("transmissionDateFrom"),
                "to": src.get("transmissionDateTo"),
            })
        fetched += len(entries)
        since = data.get("since")
        if not since or len(entries) < params["limit"]:
            break
    return rows

EVENT_PAGE = 50  # GFW v3 /events caps limit at 50 per page

def fetch_events(session: requests.Session, start: str, end: str, limit: int = 1000) -> list[dict]:
    rows = []
    for etype, dataset in EVENT_DATASETS.items():
        body = {
            "datasets": [dataset],
            "startDate": start,
            "endDate": end,
            "geometry": SYR_POLYGON,
        }
        offset = 0
        got = 0
        while got < limit:
            page_size = min(EVENT_PAGE, limit - got)
            try:
                data = _check(session.post(f"{GFW_BASE}/events", json=body,
                                           params={"limit": page_size, "offset": offset},
                                           timeout=120))
            except requests.HTTPError as e:
                print(f"  ! {etype}: {e}")
                break
            entries = data.get("entries", []) or []
            if not entries:
                break
            for ev in entries:
                pos = ev.get("position") or {}
                ves = ev.get("vessel") or {}
                rows.append({
                    "type": etype,
                    "event_id": ev.get("id"),
                    "start": ev.get("start"), "end": ev.get("end"),
                    "lat": pos.get("lat"), "lon": pos.get("lon"),
                    "vessel_id": ves.get("id"), "vessel_name": ves.get("name"),
                    "vessel_flag": ves.get("flag"), "ssvid": ves.get("ssvid"),
                })
            got += len(entries)
            next_off = data.get("nextOffset")
            total = data.get("total")
            if next_off is None or len(entries) < page_size or (total is not None and got >= total):
                break
            offset = next_off
        print(f"  {etype:10s}: {got} events")
    return rows

def fetch_effort(session: requests.Session, start: str, end: str) -> dict:
    params = {
        "spatial-resolution": "LOW",
        "temporal-resolution": "MONTHLY",
        "datasets[0]": "public-global-fishing-effort:latest",
        "date-range": f"{start},{end}",
        "format": "JSON",
        "group-by": "FLAG",
    }
    body = {"geojson": SYR_POLYGON}
    return _check(session.post(f"{GFW_BASE}/4wings/report", json=body, params=params, timeout=120))

def write_csv(rows: list[dict], path: Path) -> None:
    import csv
    if not rows:
        path.write_text("")
        print(f"  (no rows) {path}")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {len(rows):>5} rows -> {path}")

def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch GFW data for Syria.")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--limit", type=int, default=200, help="max vessels / events per type")
    args = ap.parse_args()

    token = get_token()
    session = make_session(token)
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Fetching Syria data {args.start} -> {args.end} ...")

    print("\n[1/3] Syria-flagged vessels")
    vessels = fetch_vessels(session, limit=args.limit)
    write_csv(vessels, OUT_DIR / "syria_vessels.csv")

    print("\n[2/3] Events in the Syrian EEZ")
    events = fetch_events(session, args.start, args.end, limit=args.limit)
    write_csv(events, OUT_DIR / "syria_events.csv")

    print("\n[3/3] Apparent fishing effort (4Wings report)")
    try:
        effort = fetch_effort(session, args.start, args.end)
        (OUT_DIR / "syria_fishing_effort.json").write_text(json.dumps(effort, indent=2))
        print(f"  wrote effort report -> {OUT_DIR / 'syria_fishing_effort.json'}")
    except requests.HTTPError as e:
        print(f"  ! fishing-effort report failed: {e}")

    print("\nDone. Outputs are in the ./data/ folder.")

if __name__ == "__main__":
    main()

