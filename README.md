# RadShield AI — GEO Satellite Radiation Monitor

**Forecasting Energetic Particle Radiation Environment for Geostationary Satellites**

---

## 🌟 Overview

**RadShield AI** is a research dashboard for energetic-particle observations and experimental analysis at geostationary orbit (GEO, ~35,786 km altitude). Its available telemetry, catalogue visualization, and estimate sources are labelled in the dashboard.

## Project Metadata

| Field | Value |
|---|---|
| Project | RadShield AI |
| Event | AITHON 2.0 |
| Team | Team CodeNova |
| Team ID | 911 |
| Problem statement | Forecasting Energetic Particle Radiation Environment for Geostationary Satellites |
| Theme | AI-Powered Space Radiation Forecasting & Satellite Protection |
| Category | Software |
| Track | Space Technology & Artificial Intelligence |

The presentation describes the intended 30-minute-to-24-hour forecast and operator-support capabilities. Those are project goals, not current validated or operational capabilities: TFT inference remains disabled pending data and checkpoint provenance verification.

The TFT checkpoint currently remains **disabled for inference**. Although its parameter tensors load into the configured 10-feature architecture, the checkpoint does not contain feature names, preprocessing configuration, training-data identity, or a scaler reference. Repository training scripts disagree about input features and scaling. No forecast accuracy or operational safety performance is claimed.

The dashboard can show the existing physics-inspired radial-diffusion outputs separately from observations. They are experimental estimates, not validated forecasts or satellite-safety guidance.
The `/predict` API returns HTTP 503 until a provenance-compatible inference path is available. Storm replay evaluation also requires verified checkpoint predictions and does not emit heuristic predictions or synthetic interval scores.

---

## 📁 Repository Structure

```
.
├── radshield/
│   ├── api/
│   ├── dashboard/
│   ├── data/
│   ├── models/
│   ├── notebooks/
│   ├── serving/
│   ├── training/
│   ├── tests/
│   └── weights/
├── DataSets/
├── scripts/
├── radshield_kaggle_train.py
├── merge_pretraining.py
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Installation

Ensure Python 3.10+ is installed, then run:

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Operator Dashboard

To launch the primary interactive dashboard:

```bash
streamlit run radshield/dashboard/app.py
```

The dashboard opens in **Historical Storm Replay** so the bundled local replay data
can be explored without a live data connection. Select the live NOAA stream in the
sidebar to try the network-backed view.

The dashboard is organized around three destinations: **Overview** (conditions,
electron-flux chart, and optional screening details), **GEO fleet** (satellite
catalogue and sector estimate), and **Diagnostics** (solar-wind drivers and research
provenance). Live-feed quality and detailed measurements are available in a
collapsed disclosure. The interface uses a corporate navy-and-blue theme with
neutral surfaces, a live/replay context header, condition cards, and coordinated
light charts. Checkpoint compatibility details remain in a collapsed disclosure.
These presentation changes do not alter the underlying data, model outputs,
controls, or limitations.

The dashboard's GEO fleet view uses the [CelesTrak GP GEO TLE catalogue](https://celestrak.org/NORAD/elements/gp.php?GROUP=geo&FORMAT=tle).
It filters for near-circular, near-equatorial geostationary orbits and propagates published TLEs with SGP4;
displayed positions are calculated positions at the shown UTC time, not direct satellite telemetry.
The TLE epoch is shown separately and positions are marked stale when the elements are more than 24 hours old.
Live propagation requires network access to CelesTrak and the `sgp4` dependency from `requirements.txt`.
If the source is unavailable, the dashboard reports the error and does not substitute sample satellite positions.

### 3. Launch FastAPI Status Server

The API can be launched for its health/status and replay-list endpoints:

```bash
uvicorn radshield.api.main:app --reload --port 8000
```

Access API documentation at `http://localhost:8000/docs`. Forecast requests to `/predict` return HTTP 503 while TFT training and scaler provenance remain unresolved; this endpoint does not emit surrogate predictions.

### 4. TFT Training and Inference Status

Do not enable TFT inference or use the checkpoint for operational decisions until its training provenance, feature ordering and meanings, and preprocessing are verified against inference. The dashboard enforces this gate.

The configured `finetuned_gsat19_grasp_ulf.pth` file is a tensor-only state dictionary. Its adjacent `scaler.pkl` contains `mean` and `std` arrays without feature names or a checkpoint identifier, so the pairing cannot be verified. Two repository pipelines write this checkpoint filename with different feature schemas and scaling procedures. The required 11-year pretraining dataset referenced by the Kaggle pipeline is not included in this checkout.

The checked-in `DataSets/Kaggle_FineTuning_Dataset.csv` is available, but it contains missing target observations and lacks a data dictionary documenting the units/source of all fields. `DataSets/Kaggle_Validation_March2015.csv` has irregular timestamp intervals and was not established as an untouched test set for the configured checkpoint. These artifacts alone are insufficient to reconcile and validate the loaded weights.

To resume model work, first obtain the exact training dataset(s), data dictionary/units, preprocessing configuration, scaler-to-checkpoint provenance, and a chronological held-out test set with raw timestamp-aligned observations. Keep inference disabled until these are reconciled and tested.

The pretraining merge utility uses `RADSHIELD_DATA_DIR` to locate optional local source archives.
Without it, it looks under the repository's `DataSets/` folder. Its output can be set with
`RADSHIELD_PRETRAINING_OUTPUT`; processed storm files can be directed with
`RADSHIELD_STORM_OUTPUT_DIR`. These paths are configuration, not bundled data; missing archives
must be obtained and documented before running preparation or training.

The 11-year bootstrap utility generates synthetic, formula-based data; it does not download or
reconstruct measured NASA/NOAA telemetry. Its output is for pipeline experiments only and is not
an approved training archive or evaluation dataset. Training scripts require supplied input files
and do not substitute this generated data when source archives are missing.

Kaggle weight publishing and dashboard weight syncing use `RADSHIELD_HF_REPO` for the Hugging Face
model repository ID. Publishing also requires `HF_TOKEN`; neither integration assumes a
third-party account or repository.

### 5. Local Dataset Audit and Training Readiness

Run the read-only local CSV audit from the repository root:

```bash
python -m radshield.training.dataset_audit
```

The generated `radshield/data/historical/dataset_audit.json` records row and column counts, timestamp parsing/timezone status, ordering, duplicate timestamps, observed cadence and gaps, missing/invalid numeric values, candidate target missingness, the two conflicting training schemas, missing referenced archives, and a provisional chronological split design. The audit does not resample, interpolate, fill values, or modify datasets or weights.

The split design uses chronological 70/15/15 blocks with fully-contained input-plus-target windows and excludes windows with missing target-feature history or forecast labels. It is explicitly provisional: the `Electron_Flux` source, units and blank-value meaning are undocumented, and neither feature schema has been tied to the configured checkpoint. No training or accuracy evaluation is performed by the audit.

### 6. Dataset Provenance Verification

Generate the local-evidence-only provenance report:

```bash
python -m radshield.training.dataset_provenance
```

The generated `radshield/data/historical/dataset_provenance.json` lists each target/input field, evidence and source files, confidence/status, unresolved questions, and required actions. It also records locally absent pretraining archives and unlabelled scaler sidecars. The repository does not establish the `Electron_Flux` instrument/channel, units, source citation, or blank-value semantics. The report identifies the exact external source documentation and checkpoint-linked run manifest required before training approval. It does not enable inference or modify datasets/weights.

### 7. Radiation Dataset Source Audit

The project presentation identifies NASA CDAWeb/OMNIWeb and NOAA GOES/SWPC as research-data
references. This checkout integrates NOAA SWPC live feeds and includes local GOES/OMNI readers,
but it does not establish that the checked-in CSV target values came from those products. The local
dataset provenance and source audit below keep that distinction explicit.

The project target is **measured integral energetic-electron flux above 2 MeV**, expressed in electrons/(cm²·s·sr) (pfu), with documented 5-minute averaged observations, unambiguous UTC timestamps, satellite/instrument identity, quality flags, and preserved gaps. NOAA SWPC documents that this channel represents the outer radiation belt at GEO and cautions that >2 MeV observations can be contaminated during significant proton events.

The configured [SWPC 7-day integral-electron JSON feed](https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-7-day.json) matches the documented energy channel and cadence for live observation use. It is a rolling seven-day product, does not expose the instrument name or quality flag in the inspected records, and is **not** a historical training dataset. The [NCEI GOES 13–15 EPEAD science-quality product](https://www.ncei.noaa.gov/products/goes-1-15/space-weather-instruments) documents the E2 >2 MeV channel at one-minute cadence with science-quality processing and fill values; verify actual archive dates, per-file units/time metadata, and quality flags before use. Do not assume it produced the local `Electron_Flux` column or splice it with another instrument without cross-calibration.

The local 2017–18 `Electron_Flux` column remains unverified: no repository citation, instrument/channel, unit, upstream processing, or blank-value definition is available. Solar-wind and geomagnetic feeds are covariates and cannot replace measured electron flux as the prediction target.

Generate the machine-readable source audit with:

```bash
python -m radshield.training.radiation_dataset_source_audit
```

The report is written to `radshield/data/historical/radiation_dataset_source_audit.json` and includes source comparisons, explicit unknowns, the training-eligibility rule, and a manual validation checklist. The legacy checkpoint remains blocked from dashboard inference.

### 8. Acquire archived GOES and solar-wind CDF data

The resumable CDAWeb downloader retrieves monthly GOES-13/14/15 EPEAD E2 (>2 MeV)
science-quality CDFs, WIND SWE and MFI CDFs, and five-minute OMNI near-Earth solar-wind
predictors. It preserves satellite identifiers, GOES proton-contamination quality flags,
WIND plasma quality flags, and OMNI source/interpolation fields. GOES flux is admitted as a
measurement only when its quality flag is zero and its value is finite and positive. Missing
measurements are not filled.

Request the nominal 2010–2020 archive (the end date is exclusive):

```bash
python -m radshield.data.cdaweb_archive --start 2010-01-01 --end 2021-01-01
```

Downloads and normalized CSVs are saved under the ignored local directory
`radshield/data/archive/`; `archive_manifest.json` records per-month source requests, data
availability, file hashes, and parse errors. The requested date span is not a guarantee that
every GOES spacecraft has observations throughout it; use the manifest to inspect actual
coverage. The 2024 storm benchmark CSVs and the existing `Electron_Flux` CSV are not substituted
for missing archive measurements.

### 9. Train and evaluate an archive-specific TFT

After archive acquisition, train a satellite-specific experimental model using the official
quality-flagged GOES target and time-shifted OMNI predictors:

```bash
python -m radshield.training.train_cdaweb_tft --satellite GOES-15
```

Training windows require a complete 24-hour input and 12-hour measured target at five-minute
cadence; target gaps and incomplete predictor windows are excluded, not interpolated. The split
is chronological (70% train, 15% validation, 15% held out for test), and the scaler is fitted
using training windows only. The default hourly window stride provides more training examples
than a non-overlapping daily stride, but the resulting windows overlap and their test metrics are
not independent event samples. Results, quantile metrics at +30 minutes, +6 hours and +12 hours,
checkpoint/scaler hashes, hashes of the source archive manifest, prepared dataset and training
script, random seed, and exact feature schema are saved alongside the model under
`radshield/weights/archive_tft/`.

These artifacts are research-evaluation outputs, not operational approval. They are
satellite-specific, and no checked-in, provenance-verified GSAT-19/GRASP series is available
for Indian-longitude comparison. The new training utility does not unlock or replace the
legacy dashboard/API inference guard.

### 10. Live NOAA radiation observation panel

In live mode, the dashboard displays only the unfilled `observed_flux` series from the existing
[NOAA SWPC GOES integral-electron 7-day feed](https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-7-day.json).
It reports the observation and fetch timestamps, feed state, dashboard freshness, valid observation
count and coverage, the satellite IDs present in the response, and source/channel metadata. The feed
does not provide the instrument name or an event-quality flag; these remain explicitly unavailable.
Missing target samples stay as gaps, and the panel never substitutes the model-ready filled `flux`
values for observations.

The displayed freshness state uses a 15-minute dashboard criterion for this nominally five-minute
product; it is not an official NOAA quality flag. NOAA warns that the >2 MeV channel may be invalid
during significant proton events. Because this feed has no event-quality flag, that validity remains
unknown. The source is a rolling seven-day product, not a long-term training archive. These data
quality messages are separate from the experimental project flux screening; neither is official
NOAA alerting or satellite-safety guidance. The panel reuses the dashboard's existing NOAA fetch and
does not create a second request path.

The live solar-wind index aligns NOAA's 3-hour Kp records to the five-minute telemetry timeline;
missing Kp intervals use an explicitly estimated solar-wind fallback. Live Dst and AE values are
estimates, not direct measurements. Dynamic pressure, coupling electric field, and total magnetic
field are derived from available inputs; fixed or assumed model-input values remain documented in
the sidebar and must not be read as direct observations.
