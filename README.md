# GFW + enrichment edition - Syria

`vessel_anomaly_detection.ipynb` = the real-data GFW notebook **plus
four enrichment layers applied to the flagged/suspicious vessels only**.

Sections 1–14 are the unchanged GFW pipeline (pull Syria events → features →
Isolation Forest + LOF + rules → `high_risk`). Sections 15–16 add enrichment.

## The 4 layers (run on the suspect subset)

| # | Source | Token | Scope | Adds |
|---|---|---|---|---|
| 1 | **VesselAPI** | hidden prompt | Top-N (free plan ≈150/mo → `ENRICH_MAX_VAPI`, default 25) | live identity (imo, type, callsign) |
| 2 | **DataDocked** | hidden prompt | all suspects | particulars by MMSI (imo, type, flag, owner, year, length) |
| 3 | **OFAC SDN** | none (free) | all suspects | `sanctioned` flag + `ofac_program` (match by IMO + name) |
| 4 | **Equasis** | manual CSV | all suspects | registered owner / manager / class / P&I |

**Detect broad, enrich narrow.** Free/bulk layers (OFAC, DataDocked) cover every
suspect; the metered VesselAPI is capped. Every API response is **cached to disk**
(`outputs/enrich_cache/`) so re-runs cost **0** calls.

## Hidden token inputs

GFW (Section 2) and VesselAPI / DataDocked (Section 15) all use **masked
`getpass` prompts** - nothing is echoed. Leave a prompt blank to skip that layer.

## Key design facts

- **Join key:** GFW events give MMSI (`ssvid`); IMO is resolved from
  DataDocked/VesselAPI, then used for OFAC-by-IMO and Equasis.
- **Budget reality:** enriching ~220 vessels through VesselAPI's free 150/month
  plan is impossible, so VesselAPI is capped while OFAC (free) and DataDocked
  (bulk) cover all suspects.
- **Equasis has no public API** and its terms restrict automated extraction, so
  the notebook **does not scrape it**. It writes `outputs/equasis_lookup_TODO.csv`
  (the flagged IMOs); you look them up on Equasis, save
  `outputs/equasis_filled.csv` (`imo,registered_owner,manager,class_society,pi_club`),
  and re-run the Equasis cell to merge.
- **OFAC** screening is a *screening aid*, not legal advice - confirm any hit
  against the official SDN entry. (OpenSanctions is a good fuzzy/EU/UN alternative.)

## Outputs (`./outputs/`)

| File | What |
|---|---|
| `syria_suspects.csv` | the flagged subset (pre-enrichment) |
| `syria_enriched_suspects.csv` | suspects + all layers, re-ranked by `enriched_risk` |
| `equasis_lookup_TODO.csv` | IMOs to look up on Equasis |
| `syria_enriched_dashboard.html` | map: SANCTIONED / High-risk / Other layers, owner+sanctions tooltips |

## Enriched risk

`enriched_risk = risk_score + 5×sanctioned + 0.5×(missing IMO)` - sanctions
dominate the ranking; tune the weights in the combine cell.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook vessel_anomaly_detection.ipynb
# enter GFW token (hidden), run through scoring, then enter VesselAPI/DataDocked
# tokens (hidden) for the enrichment layers.
```

> Not executed in CI (GFW/VesselAPI/DataDocked hosts are blocked by the build
> environment's egress allowlist). All non-API logic - suspect selection, OFAC
> parsing/screening, merges, re-ranking, enriched dashboard - was unit-tested
> offline against mock data.
