# Phase 24A — CelesTrak and SGP4 Integration Report

## Audit findings

The inspected `BAH-main` checkout did not contain a CelesTrak fetcher, an
`sgp4` dependency, orbital propagation, coordinate conversion, a GEO catalogue,
or a Plotly orbital globe. The existing Streamlit dashboard and its Plotly
chart helper were present. The existing tests covered the radiation-model
pipeline, not orbital data.

The checkout has no `.git` metadata, so `git status` could not report a
worktree state. No existing files were reverted.

## Changes made

- Added `kavach/data/orbital.py` as the single CelesTrak/SGP4 implementation.
  It fetches CelesTrak's configured public `GROUP=geo` TLE feed, with a
  `(3.05, 10)` second connect/read timeout, and handles HTTP, network, timeout,
  empty-feed, and malformed-record outcomes explicitly.
- The parser preserves the satellite name and NORAD identifier, validates line
  structure, checksums, matching identifiers, and SGP4-compatible fields, and
  records the TLE epoch separately from the UTC fetch timestamp. It rejects
  bad records while preserving other valid records in a mixed response.
- Propagation uses `sgp4` WGS-72 and an explicit, timezone-aware UTC instant.
  SGP4 error codes and non-finite coordinates do not become plotted positions.
  Each successful result retains its source TLE and propagation timestamp.
- TEME positions in kilometres are rotated into pseudo-Earth-fixed coordinates
  using Greenwich mean sidereal time. Longitude is wrapped to `[-180, 180)`;
  latitude is explicitly reported as geocentric.
- Added a seven-day TLE-age status label. This is an operational staleness
  indicator, not a positional-accuracy guarantee.
- Added an interactive orthographic Plotly globe and inspectable satellite
  metadata table to the existing Streamlit dashboard. The dashboard caches
  catalogue requests for 15 minutes and persists each successfully validated
  TLE response outside the repository. During an outage it reuses saved TLEs
  with their original fetch timestamp and TLE epoch, labeling their age; if no
  successful catalogue is available, it still displays an Earth-only globe
  with no markers and offers an explicit retry button.
- Declared `sgp4>=2.23` in both dependency manifests. The existing prediction
  route and model assets were not changed.

## Files changed

- `kavach/data/orbital.py` — ingestion, validation, persistent cache,
  propagation, coordinate conversion, status, and globe implementation.
- `kavach/dashboard/app.py` — orbital panel, outage recovery, propagation,
  globe, and metadata display.
- `kavach/tests/test_orbital.py` — deterministic parser, fetch, cache recovery,
  propagation, coordinate, stale-status, and plotted-marker tests.
- `requirements.txt` — added the SGP4 dependency.
- `kavach/requirements.txt` — added the SGP4 dependency.
- `PHASE_24A_CELESTRAK_SGP4_REPORT.md` — this report.

## Verification

### Mocked/unit tests

Command, run from the `BAH-main` directory:

```powershell
& '..\.venv\Scripts\python.exe' -m unittest discover -s kavach\tests -v
```

Result: **26 tests passed**. The suite includes mocked successful CelesTrak
responses, timeout, HTTP failure, empty response, valid and malformed TLEs,
successful TLE persistence and timeout recovery, SGP4 success/error/non-finite
cases, UTC enforcement, TEME rotation and longitude wrapping, stale TLE status,
Earth-only outage view, and equality between catalogue propagations and plotted
marker coordinates. Mocked success cases use isolated temporary cache paths so
test TLE fixtures cannot leak into the application's persistent catalogue.

### Live connectivity and dashboard smoke check

- Live CelesTrak connectivity was verified separately from mocked tests on
  2026-10-23. The configured GEO endpoint returned **568 valid TLE records**;
  all 568 parsed and none were rejected. The successful fetch timestamp was
  `2026-10-23T05:17:33.689524Z`.
- The running local Streamlit dashboard was checked at
  `http://127.0.0.1:8501/` and returned HTTP 200.
- Browser verification of the orbital panel confirmed **568 plotted
  markers**, a TLE-epoch metadata column, and no catalogue-unavailable or SGP4
  propagation-failure message.
- Follow-up outage check on 2026-10-23: the configured endpoint and the
  `www.celestrak.org` and legacy `celestrak.com` host variants could not
  establish a network connection within 20 seconds. No prior successful
  catalogue was present in the local persistent cache at that time, so there
  were no genuine TLEs available to plot during this particular outage.
- Persistent-cache recovery was verified with mocked network responses:
  fetching a valid TLE writes a cache, and a later mocked timeout returns that
  same TLE with its original fetch timestamp and the new failure status.
- A test-fixture TLE was found in the application cache after an earlier
  mocked test had written to the default cache location. That test now uses an
  isolated temporary cache, and the confirmed Vanguard 1 fixture cache was
  removed. Cache schema/source metadata is validated before reuse.
- After the follow-up dashboard fix, the local page rendered an Earth globe
  during the timeout without a Python exception. Satellite markers remained
  unavailable because neither live nor saved TLE records were accessible.

## Limitations

- TEME-to-Earth-fixed conversion uses the SGP4 library's GMST routine with UTC
  as an approximation to UT1. Polar motion and higher-precision frame
  transformations are not applied.
- The globe uses geocentric latitude, not WGS-84 geodetic latitude. Displayed
  markers are sub-satellite points; orbital altitude is not drawn to scale.
  Coordinates are orbital-position estimates, not precision geodesy.
- TLEs lose predictive quality with age. The seven-day stale label does not
  estimate error bounds or guarantee a position accuracy.
- A first launch needs one successful CelesTrak fetch before real satellite
  markers can be shown. If fetching fails and no valid persisted GEO catalogue
  exists, the app displays an Earth-only globe rather than fabricated
  positions.
- This integration provides orbital-position estimates only. It does not
  provide radiation forecasts, satellite-safety recommendations, or
  TLE-age-dependent accuracy claims.
