# Super Typhoon Bavi — AI Weather Model Forecast, July 2026

![SFNO and AIFS2 forecast tracks of Super Typhoon Bavi](docs/images/bavi_track_comparison.png)

A 120-hour, 6-hourly forecast of Super Typhoon Bavi from the ECMWF IFS operational analysis of 2026-07-06 12 UTC, aimed at its expected landfall near Wenzhou, China (28.0°N, 120.7°E). NVIDIA's SFNO with the Wu–Duan cyclone tracker forecasts the storm's path and intensity, and ECMWF's AIFS2 and Google DeepMind's GraphCast forecast its rain and 10 m wind; the SFNO track is then verified against the IBTrACS best track. Everything runs with [NVIDIA Earth2Studio](https://github.com/NVIDIA/earth2studio).

## Results at a glance

The SFNO track (Wu–Duan tracker) is scored against the IBTrACS best track. IBTrACS positions and pressures are agency analyses, and the tracker defines the storm center differently, so some position error is definitional. Precipitation and wind from AIFS2 and GraphCast are shown as forecast products and are not scored. For 2026 storms IBTrACS still holds provisional tracks, which can change when the agencies release their final best tracks. The forecast ends 120 h after initialization; if Bavi's observed closest approach to Wenzhou comes later than that, the timing error is limited by the forecast length (compare the last two columns of the second table).

**Track and intensity error by lead time**

<!-- bavi_track_metrics:START -->
_Filled in after the verified run._
<!-- bavi_track_metrics:END -->

**Closest approach to Wenzhou**

<!-- bavi_landfall_metrics:START -->
_Filled in after the verified run._
<!-- bavi_landfall_metrics:END -->

![SFNO forecast minimum sea-level pressure and maximum 10 m wind of Bavi by lead time](docs/images/bavi_intensity.png)
_SFNO / Wu–Duan intensity: minimum sea-level pressure (inverted axis) and maximum 10 m wind along the forecast track._

![AIFS2 and GraphCast storm-total precipitation maps for Bavi](docs/images/bavi_precip_total.png)
_Storm-total precipitation over the 120 h forecast, AIFS2 (left) and GraphCast (right): red is light rain, blue-purple is heavy rain._

![AIFS2 and GraphCast peak 10 m wind maps for Bavi](docs/images/bavi_wind_peak.png)
_Highest 10 m wind at each point over the forecast (km/h), AIFS2 and GraphCast._

![AIFS2 and GraphCast 10 m wind maps at the expected landfall time](docs/images/bavi_wind_landfall.png)
_10 m wind at the expected landfall time set in the notebook (2026-07-11 00 UTC), AIFS2 and GraphCast._

## Models and data

| Model | Developer | Initial condition | Used for |
|---|---|---|---|
| SFNO | NVIDIA | ECMWF IFS operational analysis, 2026-07-06 12 UTC | Storm track and intensity (Wu–Duan tracker), scored against IBTrACS |
| AIFS2 | ECMWF | Same IFS analysis | Precipitation and 10 m wind; a second Wu–Duan track for comparison |
| GraphCast (operational) | Google DeepMind | Same IFS analysis | Precipitation and 10 m wind |
| IBTrACS | NOAA NCEI (agency best tracks) | — | Best-track truth for the SFNO track (position and minimum pressure) |

## Quick start

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and an NVIDIA GPU with 46 GB+ of VRAM.

```bash
git clone https://github.com/ChristianByars/Bavi.git
cd Bavi
uv sync
uv run python -m ipykernel install --user --name bavi --display-name "bavi"
```

Open a notebook in JupyterLab and select the **bavi** kernel, then run all cells.

## Notebooks

<!-- RUNS:START -->
_Runtime and GPU memory are filled in after the verified run._
<!-- RUNS:END -->

| Notebook | What it does |
|---|---|
| `bavi_forecast.ipynb` | Runs SFNO with the Wu–Duan tracker, AIFS2 (wind and rain fields plus a second track) and GraphCast from one IFS analysis; maps daily and storm-total rain and 10 m wind; summarizes the forecast at Wenzhou; scores the SFNO track against IBTrACS and writes the two tables above to `docs/results/`. |

## Repository layout

```text
Bavi/
├── bavi_forecast.ipynb   # forecast and verification notebook
├── ibtracs.py            # IBTrACS best-track loader and track verification
├── metrics.py            # forecast-skill metrics shared by the tutorial repos
├── tests/                # pytest unit tests for metrics.py and ibtracs.py
├── docs/
│   ├── environment.md    # pinned versions, install caveats and fixes
│   ├── images/           # README figures
│   └── results/          # metric tables (CSV) written by the notebook
├── pyproject.toml        # dependencies (uv)
├── uv.lock               # exact tested versions
├── figures/              # every figure the notebook saves (git-ignored)
└── outputs/bavi/         # Zarr forecasts, extracted tracks, cached IBTrACS file (git-ignored)
```

## Environment notes

Pinned versions, install caveats, and fixes are in [docs/environment.md](docs/environment.md).

## References

- [NVIDIA Earth2Studio](https://github.com/NVIDIA/earth2studio) and its [0.20.0 documentation](https://nvidia.github.io/earth2studio/0.20.0/), including the [cyclone-tracking example](https://nvidia.github.io/earth2studio/0.20.0/examples/02_medium_range/05_cyclone_tracking/) this notebook follows
- [IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive): International Best Track Archive for Climate Stewardship (NOAA NCEI)
- Wu and Duan (2023), the tropical-cyclone detection method behind `TCTrackerWuDuan`: [doi:10.1016/j.wace.2023.100626](https://doi.org/10.1016/j.wace.2023.100626)
- [uv](https://docs.astral.sh/uv/), the Python package manager used for the environment
