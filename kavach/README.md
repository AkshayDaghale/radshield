# RadShield AI — Forecasting Energetic Particle Radiation at GEO

**Original development: Bharatiya Antariksh Hackathon 2026 | Team DigiIndia**  
*Team Leader:* Yashika Soni  
*Members:* Supriyo Chowdhury · Priyanshi Trivedi · Swayam Maitra  
*Original challenge:* PS-14 ISRO — Prediction of Energetic Particle Fluxes (>2 MeV) at Geostationary Orbit

---

## 🌟 Overview

**RadShield AI** is a hybrid machine-learning and physics project for forecasting energetic electron particle fluxes (>2 MeV) at geostationary orbit (GEO, ~35,786 km altitude), supporting assessment of deep-dielectric charging hazards to satellites.

### Key Capabilities & Objectives
- **Multi-Horizon Forecasts:**
  - **T+30 to 45 min:** Operational warning window (Mandatory Minimum)
  - **T+6 hr:** Medium-range satellite operations planning
  - **T+12 hr:** Extended mission planning
- **Dual-Engine Architecture:**
  - **Engine 1 (ML):** Temporal Fusion Transformer (TFT) with multi-quantile probabilistic output ($P_{10}, P_{25}, P_{50}, P_{75}, P_{90}$).
  - **Engine 2 (Physics):** 1-D Radial Diffusion solver integrating the Brautigam & Albert empirical diffusion coefficient ($D_{LL} = 10^{-9.325 + 0.506 K_p} L^{10}$) and $K_p$-dependent electron loss timescale ($\tau_{loss}$).
- **Regime-Aware Ensemble & Risk Translation:**
  - Dynamic weighting based on magnetospheric state (*Quiet, Storm Onset, Main Phase, Recovery*).
  - ML vs. Physics agreement ratio utilized as a physically grounded uncertainty score.
  - Actionable **Green / Yellow / Red** Deep-Dielectric Charging Risk Index.
- **ISRO GSAT-19 Domain Adaptation:**
  - Solves local data scarcity by pre-training on 11 years of GOES data and fine-tuning on GSAT-19 GRASP payload data at $48^\circ\text{E}$ Indian longitude sector.
- **Precursor Signal Advantage:**
  - Incorporates INTERMAGNET Hyderabad (HYB) ground magnetometer Pc5 ULF wave power (1.7–6.7 mHz), providing a critical 30-minute precursor warning before flux enhancements.

---

## 📁 Repository Structure

```
kavach/
├── data/
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
│   ├── train_tft.py            # Training script with storm-weighted loss
│   ├── finetune_grasp.py       # Transfer learning on GSAT-19 GRASP
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

### 3. Launch FastAPI REST Server

To serve predictions programmatically for satellite ground systems:

```bash
uvicorn api.main:app --reload --port 8000
```

Access API documentation at `http://localhost:8000/docs`.

### 4. Run Model Training & Evaluation

To train the TFT model locally:

```bash
python -m kavach.training.train_tft
```

To run transfer learning on GSAT-19 GRASP data:

```bash
python -m kavach.training.finetune_grasp
```

To evaluate metrics across historical storm replays:

```bash
python -m kavach.training.evaluate
```

---

## 📊 Verification Metrics Summary

| Metric | Target | Previously Reported Result | Interpretation |
| :--- | :--- | :--- | :--- |
| **Log-RMSE** | $< 0.5$ | **0.38** | Primary log-space accuracy |
| **Threat Score ($TS$)** | $\ge 0.40$ | **0.52** | Storm detection skill at $>10^4$ pfu |
| **False Alarm Rate ($FAR$)** | $< 0.30$ | **0.19** | Operator trust |
| **PICP (90% Band)** | $\ge 85\%$ | **89.2%** | Uncertainty calibration |
| **Persistence Skill** | $> 0.20$ | **0.28** | Outperforms naive baseline |
| **GRASP Domain Adaptation** | $> +15\%$ | **+21.4%** | Effective transfer learning |

---

## Original Development Team — DigiIndia
- **Yashika Soni** (Team Leader) — Modern Institute of Technology and Research Centre, Alwar
- **Supriyo Chowdhury** — University College of Engineering and Technology, Hazaribagh
- **Priyanshi Trivedi** — Presidency University, Bangalore
- **Swayam Maitra** — Institute of Engineering and Management, Kolkata
