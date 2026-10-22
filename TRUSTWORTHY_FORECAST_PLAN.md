# Making RadShield Forecasts Trustworthy

## Current status

RadShield can now acquire and prepare measured GOES and near-Earth solar-wind
observations from NASA SPDF CDAWeb, train a satellite-specific TFT, preserve
source and artifact hashes, and produce a 12-hour, five-quantile output. That
proves the training pipeline executes; it does **not** prove the predictions
are useful or safe to act on. Keep the dashboard and API TFT inference gate
closed.

The current GOES-15 experiment used 191 training, 182 validation, and 102
chronological held-out windows. Its held-out median absolute error (MAE) was
worse than persistence at every requested horizon:

| Horizon | TFT MAE (log10 PFU) | Persistence MAE (log10 PFU) | P10–P90 coverage |
|---|---:|---:|---:|
| 30 minutes | 1.495 | 0.076 | 3.9% |
| 6 hours | 2.602 | 0.447 | 0% |
| 12 hours | 1.921 | 0.578 | 7.8% |

An 80% prediction interval should cover approximately 80% of observations over
a representative evaluation set. These measured results show substantial
forecast error and severe interval under-coverage. The model is not currently
trustworthy.

The downloaded GOES-15 E2 archive covers March 2010 through March 2020, not
eleven complete years. Other GOES spacecraft have different coverage and
instrument histories. OMNI HRO is time-shifted near-Earth data assembled from
multiple spacecraft; it must not be described as WIND-only. No provenance-
verified GSAT-19/GRASP target series is currently available in the project.
The source coverage and exact files used are recorded in
`radshield/data/archive/archive_manifest.json`; the GOES-15 run details and
artifact hashes are in `radshield/weights/archive_tft/goes15_manifest.json`.

## Work required, in order

### 1. Lock down the prediction target and data contract

- Keep the target explicit: GOES EPEAD E2 integral electron flux above 2 MeV,
  in electrons/(cm² s sr), with only documented good-quality measurements.
- Document each source, instrument, variable, unit, quality flag, fill value,
  timestamp convention, cadence, and any processing applied.
- Preserve missing observations as missing. Do not interpolate target labels or
  present filled values as measurements.
- Record OMNI source-code and interpolation metadata. Decide and document
  whether multi-spacecraft OMNI or direct WIND inputs are scientifically
  intended; they are not interchangeable claims.
- Keep missions separate until cross-calibration and instrument transitions
  have been assessed. Do not silently concatenate GOES satellites into one
  target series.
- For Indian-sector evaluation, obtain time-aligned, quality-controlled
  GSAT-19/GRASP measurements with a data dictionary, calibration/quality
  metadata, units, and permission to use the data. GOES is not a substitute for
  the requested GSAT-19 validation.

### 2. Make the experiment reproducible and leakage-resistant

- Freeze the source archive, preprocessing, feature order, model configuration,
  random seeds, and checkpoint/scaler pairing in a run manifest. Keep hashes
  for the raw inputs and prepared dataset.
- Verify every feature is available at the forecast issue time. Audit
  timestamp alignment, duplicate samples, cadence gaps, fill values, and any
  information that could leak from the future.
- Keep preprocessing statistics fitted on training data only. Never choose
  features, hyperparameters, epochs, or thresholds using the final test set.
- Split chronologically and reserve complete later storm/event periods for
  testing. Report event-level results and uncertainty (for example,
  event-block bootstrap intervals), not only scores over densely overlapping
  five-minute windows.
- Preserve the existing strict complete-window policy. The hourly window
  stride increases the number of examples but produces overlapping windows;
  it does not create independent storms or independent evidence.

### 3. Establish meaningful baselines and diagnose the current failure

- Compare each horizon against persistence, a training-only climatology, and
  at least one simple time-series baseline. Report the same metrics on exactly
  the same held-out timestamps.
- Use validation data to diagnose the large low-biased predictions seen in the
  current run. Inspect target scaling, target distributions across time,
  instrument changes, missingness, feature/target alignment, and model capacity
  before changing architecture.
- Recheck quantile ordering and calibration. Do not sort crossed quantiles
  only for scoring; report crossing frequency and evaluate the actual deployed
  quantile outputs.
- Tune only on training/validation data. Retrain with recorded configuration,
  then evaluate once on untouched event blocks.

### 4. Require forecast skill and uncertainty calibration

Before deployment, pre-register acceptance criteria and require all of the
following on untouched, representative event blocks:

- Better error and quantile loss than the chosen baselines at **each** supported
  horizon, with uncertainty intervals showing the improvement is not merely
  sampling noise.
- Calibrated prediction intervals: P10–P90 coverage should be close to its
  nominal 80% on the full test set and reported separately for quiet,
  disturbed, and extreme conditions. Set numerical tolerances before test
  evaluation.
- Useful performance across events, seasons, data gaps, and instrument eras;
  disclose failures and sample counts rather than averaging them away.
- Stable results from repeated seeded runs and a documented model/scaler/
  feature-schema compatibility check.
- Independent reproduction from the archived raw data and the recorded run
  manifest.

If the model cannot beat persistence reliably, do not publish its output as a
forecast. Keep showing observations and clearly labelled experimental analysis
instead.

### 5. Validate operations separately from historical accuracy

- Run a live shadow period that records issue-time inputs and predictions
  without influencing operator decisions. Compare them later with quality-
  controlled observations.
- Monitor input freshness, missingness, source changes, feature drift,
  prediction range, quantile crossings, and calibration drift. Fail closed and
  show a clear status when requirements are not met.
- Display forecast horizon, issue time, target definition, source, uncertainty,
  freshness, and validation status together. Never label simulated,
  interpolated, radial-diffusion, or synthetic preview data as TFT output.
- Require domain review and a documented rollback/disable procedure before any
  operational use. A historical metric alone is not a satellite-safety
  approval.

## Practical next steps

1. Preserve the current checkpoint as a research artifact; do not replace the
   dashboard/API inference guard with it.
2. Review the run manifest and target/predictor quality rules with a space-
   weather or radiation-effects domain reviewer.
3. Acquire provenance-verified GSAT-19/GRASP measurements if Indian-sector
   target validation remains a project requirement.
4. Diagnose the held-out failure using validation data and compare against
   persistence and climatology before spending time on larger models.
5. Repeat the pre-registered, event-blocked evaluation and only then decide
   whether the forecast gate can be reconsidered.

## Current verification snapshot

- Archive manifest: 640 downloaded, 5 cached, 147 no-data intervals; no
  remaining download or parse errors after retrying two timed-out requests.
- Project test suite: 51 tests passed.
- Trained GOES-15 checkpoint: finite output shape `(1, 144, 5)`; this is a
  runtime-shape check, not a skill claim.
- Local Streamlit dashboard: `/` and `/_stcore/health` responded successfully
  during the latest check.
- Live NOAA feeds were unavailable during browser verification; the dashboard
  displayed an explicit unavailable-data notice rather than substituting live
  values.
- The GEO catalogue view showed all 472 loaded TLE positions as stale (>24
  hours). Treat these positions as stale, not current tracking.
- The research/provenance view explicitly labelled the older stored preview
  arrays as synthetic demonstrations, not TFT predictions or verified
  performance metrics.
