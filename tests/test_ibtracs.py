# tests/test_ibtracs.py
from datetime import datetime

import pandas as pd
import pytest

import ibtracs

# A tiny copy of the IBTrACS v04r01 CSV layout: the real files have the same header row, then a units row
# (row 1, skipped by the loader), and write missing values as a single blank.
HEADER = "SID,SEASON,BASIN,NAME,ISO_TIME,LAT,LON,WMO_PRES,TRACK_TYPE,USA_PRES"
UNITS = " ,Year, , , ,degrees_north,degrees_east,mb, ,mb"
ROWS = [
    "2026182N09163,2026,WP,BAVI,2026-07-06 12:00:00,20.0,140.0, ,PROVISIONAL,990",
    "2026182N09163,2026,WP,BAVI,2026-07-06 15:00:00,20.5,139.5, ,PROVISIONAL,988",  # 3-hourly row: must be dropped
    "2026182N09163,2026,WP,BAVI,2026-07-06 18:00:00,21.0,139.0,985,PROVISIONAL, ",  # falls back to WMO_PRES
    "2026182N09163,2026,WP,BAVI,2026-07-06 18:45:00,21.1,138.9, ,PROVISIONAL,984",  # off-hour landfall fix: dropped
    "2026190N10150,2026,WP,BAVI,2026-07-06 12:00:00,19.0,141.0, ,US-PROVISIONAL_spur,995",  # spur track: dropped
    "2026183N10100,2026,WP,OTHER,2026-07-06 12:00:00,10.0,100.0, ,PROVISIONAL,1000",
    "2025187N10100,2025,WP,BAVI,2025-07-06 12:00:00,10.0,100.0, ,main,1000",
]


def _csv(rows):
    return "\n".join([HEADER, UNITS, *rows]) + "\n"


CSV = _csv(ROWS)


def test_load_best_track_filters_storm_and_synoptic_times(tmp_path):
    path = tmp_path / "ibtracs.csv"
    path.write_text(CSV)
    best = ibtracs.load_best_track("Bavi", 2026, csv_path=path)
    assert best["valid_time"].dt.hour.tolist() == [12, 18]
    assert best["mslp_hpa"].tolist() == [990.0, 985.0]
    assert best["lon"].tolist() == [140.0, 139.0]
    assert list(best.columns) == ["valid_time", "lat", "lon", "mslp_hpa"]


def test_load_best_track_unknown_storm_raises(tmp_path):
    path = tmp_path / "ibtracs.csv"
    path.write_text(CSV)
    with pytest.raises(ValueError):
        ibtracs.load_best_track("Nobody", 2026, csv_path=path)


def test_load_best_track_joins_consecutive_segments_across_the_dateline(tmp_path):
    # IBTrACS can store one storm as several SIDs (for example after it crosses the dateline), and its
    # longitudes may be given in -180..180 or beyond 180; the loader joins the pieces and returns 0-360.
    rows = [
        "2024247N28179,2024,WP,HONE,2024-09-03 00:00:00,28.0,179.4, ,main, ",
        "2024235N16223,2024,EP,HONE,2024-09-02 12:00:00,27.0,-178.0, ,main,1000",
        "2024235N16223,2024,EP,HONE,2024-09-02 18:00:00,27.5,184.5, ,main,1001",
    ]
    path = tmp_path / "ibtracs.csv"
    path.write_text(_csv(rows))
    best = ibtracs.load_best_track("hone", 2024, csv_path=path)
    assert best["valid_time"].tolist() == list(pd.to_datetime(["2024-09-02 12:00", "2024-09-02 18:00", "2024-09-03 00:00"]))
    assert best["lon"].tolist() == pytest.approx([182.0, 184.5, 179.4])
    assert best["mslp_hpa"].iloc[:2].tolist() == [1000.0, 1001.0] and pd.isna(best["mslp_hpa"].iloc[2])


def test_load_best_track_overlapping_segments_raise(tmp_path):
    rows = [
        "2026231N08155,2026,WP,SAUDEL,2026-08-20 00:00:00,15.0,140.0, ,PROVISIONAL,990",
        "2026232N09150,2026,WP,SAUDEL,2026-08-20 00:00:00,16.0,135.0, ,PROVISIONAL,995",
    ]
    path = tmp_path / "ibtracs.csv"
    path.write_text(_csv(rows))
    with pytest.raises(ValueError, match="overlap"):
        ibtracs.load_best_track("Saudel", 2026, csv_path=path)


def test_load_best_track_reads_the_cached_file_without_downloading(tmp_path, monkeypatch):
    (tmp_path / ibtracs.IBTRACS_LAST3YEARS_URL.rsplit("/", 1)[-1]).write_text(CSV)

    def no_network(url, dest):
        raise AssertionError("must not download when the cached file exists")

    monkeypatch.setattr(ibtracs, "urlretrieve", no_network)
    best = ibtracs.load_best_track("Bavi", 2026, cache_dir=tmp_path)
    assert len(best) == 2


def test_load_best_track_downloads_once_and_leaves_no_partial_file(tmp_path, monkeypatch):
    calls = []

    def fake_urlretrieve(url, dest):
        calls.append(url)
        with open(dest, "w") as f:
            f.write(CSV)

    monkeypatch.setattr(ibtracs, "urlretrieve", fake_urlretrieve)
    cache = tmp_path / "cache"
    for _ in range(2):
        assert len(ibtracs.load_best_track("Bavi", 2026, cache_dir=cache)) == 2
    assert calls == [ibtracs.IBTRACS_LAST3YEARS_URL]
    assert sorted(p.name for p in cache.iterdir()) == [ibtracs.IBTRACS_LAST3YEARS_URL.rsplit("/", 1)[-1]]


def test_verify_track_reports_position_and_pressure_errors():
    best = pd.DataFrame({
        "valid_time": pd.to_datetime(["2026-07-06 12:00", "2026-07-06 18:00"]),
        "lat": [20.0, 21.0], "lon": [140.0, 140.0], "mslp_hpa": [990.0, 985.0],
    })
    model = pd.DataFrame({
        "lead_time_h": [0, 6, 12],
        "tclat": [21.0, 22.0, 23.0], "tclon": [140.0, 140.0, 140.0], "tcmsl_hPa": [992.0, 983.0, 980.0],
    })
    out = ibtracs.verify_track(model, best, "2026-07-06 12:00")
    assert out["Lead (h)"].tolist() == [0, 6]
    assert out["Position error (km)"].tolist() == pytest.approx([111.0, 111.0], abs=1.0)
    assert out["MSLP error (hPa)"].tolist() == [2.0, -2.0]


def test_verify_track_scores_common_times_on_the_step_grid_only():
    # Second-resolution best-track times, a datetime start and an off-grid 3 h lead must all line up correctly.
    best = pd.DataFrame({
        "valid_time": pd.to_datetime(["2026-07-06 12:00", "2026-07-06 15:00", "2026-07-06 18:00", "2026-07-07 00:00"])
        .astype("datetime64[s]"),
        "lat": [20.0, 20.0, 20.0, 20.0], "lon": [140.0, 140.0, 140.0, 140.0], "mslp_hpa": [990.0, 990.0, 990.0, 990.0],
    })
    model = pd.DataFrame({
        "lead_time_h": [0, 3, 6, 12],
        "tclat": [20.0, 20.0, 20.0, 20.0], "tclon": [140.0, 140.0, 141.0, 140.0], "tcmsl_hPa": [990.0, 991.0, 1000.0, 995.0],
    })
    start = datetime(2026, 7, 6, 12)
    six = ibtracs.verify_track(model, best, start)
    assert six["Lead (h)"].tolist() == [0, 6, 12]
    assert six["MSLP error (hPa)"].tolist() == [0.0, 10.0, 5.0]
    assert six["Position error (km)"].tolist() == pytest.approx([0.0, 104.0, 0.0], abs=1.0)  # 1 deg of lon at 20 N
    assert ibtracs.verify_track(model, best, start, step_hours=12)["Lead (h)"].tolist() == [0, 12]
    empty = ibtracs.verify_track(model, best, "2030-01-01 00:00")
    assert empty.empty and list(empty.columns) == ["Lead (h)", "Position error (km)", "MSLP error (hPa)"]
