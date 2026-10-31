# Super Typhoon Bavi — Real-Time Forecast

A 5-day AI weather forecast of Super Typhoon Bavi, run with [NVIDIA earth2studio](https://github.com/NVIDIA/earth2studio).

Bavi formed NW of Guam on June 30, 2026 and reached super-typhoon intensity by ~July 2. This forecast targets its expected landfall near **Wenzhou, China** (28.0 N, 120.7 E) around July 11.

- **Init:** 2026-07-06 12Z, from a single ECMWF IFS operational analysis
- **Horizon:** 120 h (5 days), 6-hourly steps

## Models

All three start from the same IFS initial state, so they can be compared directly.

| Model | Product |
|---|---|
| SFNO | Track + intensity, via the `TCTrackerWuDuan` cyclone tracker |
| AIFS2 | Precipitation + 10 m wind |
| GraphCast (operational) | Precipitation + 10 m wind |

Because the forecast is future-valid, there is no ERA5 or best-track ground truth to verify against — these are forecast products, not skill scores.

## Setup

Requires Python 3.12 and a CUDA-capable GPU (~46 GB VRAM).

```bash
uv sync
```

## Running

Open `bavi_forecast.ipynb` in JupyterLab using the `bavi` kernel and run all cells, or run it headless:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace bavi_forecast.ipynb
```

Models are loaded and freed one at a time so only one is resident on the GPU.

## Output

- `figures/` — track, intensity, daily/total precipitation, and peak/landfall wind maps
- `outputs/bavi/` — raw forecast fields (Zarr) and extracted storm tracks
