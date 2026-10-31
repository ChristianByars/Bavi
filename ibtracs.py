"""IBTrACS best-track loading and forecast-track verification for the Bavi notebook.

IBTrACS (NOAA NCEI's International Best Track Archive for Climate Stewardship) merges every agency's
tropical-cyclone tracks into one CSV per subset. Each CSV has a header row, then a units row, and writes
missing values as a single blank. Recent storms carry provisional tracks (TRACK_TYPE "PROVISIONAL"):
real-time operational estimates that are replaced once the agencies release their final best tracks.
"""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from metrics import haversine_km

IBTRACS_LAST3YEARS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.last3years.list.v04r01.csv"
)
_COLUMNS = ["SID", "SEASON", "NAME", "ISO_TIME", "TRACK_TYPE", "LAT", "LON", "USA_PRES", "WMO_PRES"]


def load_best_track(
    name: str, season: int, csv_path=None, cache_dir="outputs/ibtracs", url: str = IBTRACS_LAST3YEARS_URL
) -> pd.DataFrame:
    """6-hourly best track (valid_time, lat, lon in 0-360, mslp_hpa) for one named storm.

    Only synoptic fixes (00, 06, 12 and 18 UTC, on the hour) are kept: IBTrACS also lists interpolated
    3-hourly positions and off-hour landfall points. Spur tracks (alternative tracks of merging or
    splitting systems) are dropped. A storm stored as several consecutive SIDs (for example after it
    crosses the dateline) is joined; SIDs that overlap in time raise ValueError.

    Pressure is the USA agency value (JTWC in the West Pacific) with the WMO agency value as fallback;
    provisional tracks usually carry only the USA value.

    Unless `csv_path` is given, the CSV at `url` is downloaded once into `cache_dir` and reused (delete the
    cached file to refresh it). The default file covers the last three seasons; for an older storm pass
    the basin file from the same folder (for example ibtracs.WP.list.v04r01.csv) as `url`.
    """
    if csv_path is None:
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        csv_path = cache / url.rsplit("/", 1)[-1]
        if not csv_path.exists():
            partial = csv_path.with_name(csv_path.name + ".part")
            urlretrieve(url, partial)
            partial.replace(csv_path)  # an interrupted download never looks like a complete cached file
    # Row 1 holds units. Missing values are one blank; "NA" is the North Atlantic basin code, not a gap.
    df = pd.read_csv(
        csv_path, skiprows=[1], usecols=_COLUMNS, keep_default_na=False, na_values=["", " "], low_memory=False
    )
    is_storm = df["NAME"].astype(str).str.strip().str.upper() == name.strip().upper()
    is_season = pd.to_numeric(df["SEASON"], errors="coerce") == season
    is_spur = df["TRACK_TYPE"].astype(str).str.contains("spur", case=False)
    df = df[is_storm & is_season & ~is_spur]
    if df.empty:
        raise ValueError(f"no IBTrACS rows for {name!r} in season {season} in {csv_path}")
    valid_time = pd.to_datetime(df["ISO_TIME"], format="%Y-%m-%d %H:%M:%S").astype("datetime64[ns]")
    synoptic = (valid_time.dt.hour % 6 == 0) & (valid_time.dt.minute == 0) & (valid_time.dt.second == 0)
    pressure = pd.to_numeric(df["USA_PRES"], errors="coerce").fillna(pd.to_numeric(df["WMO_PRES"], errors="coerce"))
    out = pd.DataFrame({
        "valid_time": valid_time,
        "lat": pd.to_numeric(df["LAT"], errors="coerce"),
        "lon": pd.to_numeric(df["LON"], errors="coerce") % 360.0,
        "mslp_hpa": pressure,
    })[synoptic]
    out = out.dropna(subset=["lat", "lon"]).sort_values("valid_time").reset_index(drop=True)
    if out["valid_time"].duplicated().any():
        sids = sorted(df["SID"].astype(str).unique())
        raise ValueError(f"IBTrACS tracks {sids} for {name!r} in {season} overlap in time; pass a csv_path with one of them")
    return out


def verify_track(model_df: pd.DataFrame, best_df: pd.DataFrame, start_time, step_hours: int = 6) -> pd.DataFrame:
    """Per-lead position (km) and MSLP (hPa) error of a forecast track against the best track.

    `model_df` needs lead_time_h, tclat, tclon, tcmsl_hPa (one row per tracked lead); `best_df` is the output
    of `load_best_track`. Leads that are not a multiple of `step_hours`, and valid times missing from either
    track, are not scored. MSLP error is forecast minus best track: positive means the model storm is too weak.
    """
    model = model_df[model_df["lead_time_h"] % step_hours == 0].copy()
    lead = pd.to_timedelta(model["lead_time_h"], unit="h")
    model["valid_time"] = (pd.Timestamp(start_time) + lead).astype("datetime64[ns]")
    best = best_df[["valid_time", "lat", "lon", "mslp_hpa"]].copy()
    best["valid_time"] = pd.to_datetime(best["valid_time"]).astype("datetime64[ns]")  # same resolution on both sides
    joined = model.merge(best, on="valid_time", how="inner").sort_values("lead_time_h", ignore_index=True)
    return pd.DataFrame({
        "Lead (h)": joined["lead_time_h"].astype(int),
        "Position error (km)": haversine_km(joined["tclat"], joined["tclon"], joined["lat"], joined["lon"]).round(0),
        "MSLP error (hPa)": (joined["tcmsl_hPa"] - joined["mslp_hpa"]).round(1),
    })
