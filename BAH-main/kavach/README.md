# RadShield AI — GEO Satellite Radiation Monitor

**Forecasting Energetic Particle Radiation Environment for Geostationary Satellites**

---

## 🌟 Overview

**RadShield AI** is a research dashboard for energetic-particle observations and experimental analysis at geostationary orbit (GEO, ~35,786 km altitude). Its available telemetry, catalogue visualization, and estimate sources are labelled in the dashboard.

The TFT checkpoint currently remains **disabled for inference**. Although its parameter tensors load into the configured 10-feature architecture, the checkpoint does not contain feature names, preprocessing configuration, training-data identity, or a scaler reference. Repository training scripts disagree about input features and scaling. No forecast accuracy or operational safety performance is claimed.

The dashboard can show the existing physics-inspired radial-diffusion outputs separately from observations. They are experimental estimates, not validated forecasts or satellite-safety guidance.
The legacy `/predict` API returns HTTP 503 rather than serving its former formula-based surrogate as a TFT forecast. The storm replay evaluator likewise refuses to emit its former heuristic predictions and synthetic interval scores.

---

## 📁 Repository Structure

```
kavach/
├── data/
│   ├── geo_satellites.py       # CelesTrak GEO TLE parsing and SGP4 position propagation
│   ├── ingest_goes.py          # Readers for GOES-13/15/16/17/18 CDF & CSV
│   ├── ingest_omni.py          # OMNI 5-min solar wind parameter reader
│   ├── ingest_magnetometer.py  # INTERMAGNET IAGA-2002 reader + Pc5 ULF power
│   ├── ingest_grasp.py         # GSAT-19 GRASP Indian sector reader
│   ├── clean.py                # Despiking, log-transform, gap handling
│   ├── features.py             # 19 physics-informed feature vectors (Ec, Pdyn, Lags)
│   └── sample_data.py          # Synthetic dataset & storm replay generator
├── models/
│   ├── tft_model.py            # Temporal Fusion Transformer PyTorch model
│   ├── radial_diff.py          # 1-D radial diffusion ODE solver (RK45)
│   ├── regime.py               # Magnetospheric regime state classifier
│   └── ensemble.py             # Regime fusion engine + Risk classifier
├── training/
│   ├── train_tft.py            # Package fine-tuning recipe (legacy 10-feature schema)
│   ├── finetune_grasp.py       # Older synthetic-data transfer-learning example
│   ├── train_11yr_model.py     # 11-year archive training recipe
│   └── evaluate.py             # Validation suite & storm replay benchmarks
├── api/
│   └── main.py                 # FastAPI REST serving backend
├── dashboard/
│   └── app.py                  # Streamlit operator dashboard
├── weights/
│   ├── kavach_tft_v1.pt        # Model checkpoint
│   └── scaler.pkl              # Scaler instance
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_train_tft_kaggle.ipynb
│   └── 04_storm_replay_validation.ipynb
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Installation

Ensure Python 3.10+ is installed, then run:

```bash
cd kavach
pip install -r requirements.txt
```

### 2. Run the Streamlit Operator Dashboard

To launch the primary interactive dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard's GEO fleet view uses the [CelesTrak GP GEO TLE catalogue](https://celestrak.org/NORAD/elements/gp.php?GROUP=geo&FORMAT=tle).
It filters for near-circular, near-equatorial geostationary orbits and propagates published TLEs with SGP4;
displayed positions are calculated positions at the shown UTC time, not direct satellite telemetry.
The TLE epoch is shown separately and positions are marked stale when the elements are more than 24 hours old.
Live propagation requires network access to CelesTrak and the `sgp4` dependency from `requirements.txt`.
If the source is unavailable, the dashboard reports the error and does not substitute sample satellite positions.

### 3. Launch FastAPI Status Server

The API can be launched for its health/status and replay-list endpoints:

```bash
uvicorn api.main:app --reload --port 8000
```

Access API documentation at `http://localhost:8000/docs`. Forecast requests to `/predict` return HTTP 503 while TFT training and scaler provenance remain unresolved; this endpoint does not emit surrogate predictions.

### 4. TFT Training and Inference Status

Do not enable TFT inference or use the checkpoint for operational decisions until its training provenance, feature ordering and meanings, and preprocessing are verified against inference. The dashboard enforces this gate.

The configured `finetuned_gsat19_grasp_ulf.pth` file is a tensor-only state dictionary. Its adjacent `scaler.pkl` contains `mean` and `std` arrays without feature names or a checkpoint identifier, so the pairing cannot be verified. Two repository pipelines write this checkpoint filename with different feature schemas and scaling procedures. The required 11-year pretraining dataset referenced by the Kaggle pipeline is not included in this checkout.

The checked-in `DataSets/Kaggle_FineTuning_Dataset.csv` is available, but it contains missing target observations and lacks a data dictionary documenting the units/source of all fields. `DataSets/Kaggle_Validation_March2015.csv` has irregular timestamp intervals and was not established as an untouched test set for the configured checkpoint. These artifacts alone are insufficient to reconcile and validate the loaded weights.

To resume model work, first obtain the exact training dataset(s), data dictionary/units, preprocessing configuration, scaler-to-checkpoint provenance, and a chronological held-out test set with raw timestamp-aligned observations. Keep inference disabled until these are reconciled and tested.

The archive preparation utilities use `RADSHIELD_DATA_DIR` to locate optional local source archives.
Without it, they look under the repository's `DataSets/` folder. The pretraining merge output can
be set with `RADSHIELD_PRETRAINING_OUTPUT`; processed storm files can be directed with
`RADSHIELD_STORM_OUTPUT_DIR`. These paths are configuration, not bundled data; missing archives
must be obtained and documented before running preparation or training.

### 5. Local Dataset Audit and Training Readiness

Run the read-only local CSV audit from the repository root:

```bash
python -m kavach.training.dataset_audit
```

The generated `kavach/data/historical/dataset_audit.json` records row and column counts, timestamp parsing/timezone status, ordering, duplicate timestamps, observed cadence and gaps, missing/invalid numeric values, candidate target missingness, the two conflicting training schemas, missing referenced archives, and a provisional chronological split design. The audit does not resample, interpolate, fill values, or modify datasets or weights.

The split design uses chronological 70/15/15 blocks with fully-contained input-plus-target windows and excludes windows with missing target-feature history or forecast labels. It is explicitly provisional: the `Electron_Flux` source, units and blank-value meaning are undocumented, and neither feature schema has been tied to the configured checkpoint. No training or accuracy evaluation is performed by the audit.

### 6. Dataset Provenance Verification

Generate the local-evidence-only provenance report:

```bash
python -m kavach.training.dataset_provenance
```

The generated `kavach/data/historical/dataset_provenance.json` lists each target/input field, evidence and source files, confidence/status, unresolved questions, and required actions. It also records locally absent pretraining archives and unlabelled scaler sidecars. The repository does not establish the `Electron_Flux` instrument/channel, units, source citation, or blank-value semantics. The report identifies the exact external source documentation and checkpoint-linked run manifest required before training approval. It does not enable inference or modify datasets/weights.

### 7. Radiation Dataset Source Audit

The project target is **measured integral energetic-electron flux above 2 MeV**, expressed in electrons/(cm²·s·sr) (pfu), with documented 5-minute averaged observations, unambiguous UTC timestamps, satellite/instrument identity, quality flags, and preserved gaps. NOAA SWPC documents that this channel represents the outer radiation belt at GEO and cautions that >2 MeV observations can be contaminated during significant proton events.

The configured [SWPC 7-day integral-electron JSON feed](https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-7-day.json) matches the documented energy channel and cadence for live observation use. It is a rolling seven-day product, does not expose the instrument name or quality flag in the inspected records, and is **not** a historical training dataset. The [NCEI GOES 13–15 EPEAD science-quality product](https://www.ncei.noaa.gov/products/goes-1-15/space-weather-instruments) documents the E2 >2 MeV channel at one-minute cadence with science-quality processing and fill values; verify actual archive dates, per-file units/time metadata, and quality flags before use. Do not assume it produced the local `Electron_Flux` column or splice it with another instrument without cross-calibration.

The local 2017–18 `Electron_Flux` column remains unverified: no repository citation, instrument/channel, unit, upstream processing, or blank-value definition is available. Solar-wind and geomagnetic feeds are covariates and cannot replace measured electron flux as the prediction target. No local historical dataset is currently approved for training.

Generate the machine-readable source audit with:

```bash
python -m kavach.training.radiation_dataset_source_audit
```

The report is written to `kavach/data/historical/radiation_dataset_source_audit.json` and includes source comparisons, explicit unknowns, the training-eligibility rule, and a manual validation checklist. Model training remains blocked and TFT inference remains disabled.

### 8. Live NOAA radiation observation panel

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

## Original project attribution

The following team and event details are retained as historical attribution from the source project; they are not current RadShield AI branding.

- Bharatiya Antariksh Hackathon 2026, Team DigiIndia, PS-14 ISRO
- **Yashika Soni** (Team Leader) — Modern Institute of Technology and Research Centre, Alwar
- **Supriyo Chowdhury** — University College of Engineering and Technology, Hazaribagh
- **Priyanshi Trivedi** — Presidency University, Bangalore
- **Swayam Maitra** — Institute of Engineering and Management, Kolkata
