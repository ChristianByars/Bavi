"""Forecast-skill metrics (NumPy only).

This file needs nothing beyond NumPy, so it can be copied into another project as is.
Gridded fields have shape (..., lat, lon); `lat` is a 1-D array in degrees.
Area statistics are cos(latitude)-weighted and ignore NaNs.
"""
from __future__ import annotations

import numpy as np

R_EARTH_KM = 6371.0


def lat_weights(lat) -> np.ndarray:
    """cos(latitude) weights normalised to mean 1, shape (nlat,)."""
    w = np.cos(np.radians(np.asarray(lat, dtype=float)))
    return w / w.mean()


def _wmean(x, lat):
    """Area-weighted mean over the last two axes, ignoring NaNs."""
    x = np.asarray(x, dtype=float)
    w = np.broadcast_to(lat_weights(lat)[:, None], x.shape[-2:])
    w = np.broadcast_to(w, x.shape)
    ok = np.isfinite(x)
    num = np.where(ok, x * w, 0.0).sum(axis=(-2, -1))
    den = np.where(ok, w, 0.0).sum(axis=(-2, -1))
    with np.errstate(invalid="ignore", divide="ignore"):  # all-NaN input: 0/0 -> NaN, quietly
        return num / den


def weighted_bias(fc, ref, lat):
    """Mean(forecast - reference), area-weighted."""
    return _wmean(np.asarray(fc, float) - np.asarray(ref, float), lat)


def weighted_rmse(fc, ref, lat):
    """Root-mean-square error, area-weighted."""
    return np.sqrt(_wmean((np.asarray(fc, float) - np.asarray(ref, float)) ** 2, lat))


def weighted_mae(fc, ref, lat):
    """Mean absolute error, area-weighted."""
    return _wmean(np.abs(np.asarray(fc, float) - np.asarray(ref, float)), lat)


def pattern_corr(fc, ref, lat):
    """Area-weighted centred (anomaly) pattern correlation; invariant to shift and scale.

    Returns NaN (without warning) when either field has no variance over the jointly valid points,
    i.e. its anomaly variance is at most (1e-12)**2 times its mean square, or when no point is valid.
    """
    fc = np.asarray(fc, float)
    ref = np.asarray(ref, float)
    both = np.isfinite(fc) & np.isfinite(ref)
    fc = np.where(both, fc, np.nan)
    ref = np.where(both, ref, np.nan)
    fa = fc - np.asarray(_wmean(fc, lat))[..., None, None]
    ra = ref - np.asarray(_wmean(ref, lat))[..., None, None]
    var_f, var_r = _wmean(fa**2, lat), _wmean(ra**2, lat)
    tol = (1e-12) ** 2
    flat = (var_f <= tol * _wmean(fc**2, lat)) | (var_r <= tol * _wmean(ref**2, lat))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = _wmean(fa * ra, lat) / np.sqrt(var_f * var_r)
    return np.where(flat, np.nan, r)[()]


def skill_row(fc, ref, lat) -> dict:
    """Bias, RMSE, MAE and pattern correlation for one 2-D (lat, lon) field pair."""
    return {
        "Bias": round(float(weighted_bias(fc, ref, lat)), 3),
        "RMSE": round(float(weighted_rmse(fc, ref, lat)), 3),
        "MAE": round(float(weighted_mae(fc, ref, lat)), 3),
        "Pattern corr": round(float(pattern_corr(fc, ref, lat)), 3),
    }


def event_total_error(fc_total, ref_total, lat) -> dict:
    """Area-mean event totals (same units as the inputs) and the percent bias of the forecast total."""
    fc_mean = float(_wmean(fc_total, lat))
    ref_mean = float(_wmean(ref_total, lat))
    pct = 100.0 * (fc_mean - ref_mean) / ref_mean if ref_mean else float("nan")
    return {
        "Forecast area-mean": round(fc_mean, 1),
        "Reference area-mean": round(ref_mean, 1),
        "Total bias (%)": round(pct, 1),
    }


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; inputs in degrees, broadcasting, safe across the dateline."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(np.asarray(lon2, float) - np.asarray(lon1, float))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(a))


def intensity_error(p_fc, p_ref) -> dict:
    """Bias, RMSE and MAE of an intensity series (for example MSLP in hPa), ignoring NaNs."""
    d = np.asarray(p_fc, float) - np.asarray(p_ref, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"bias": float("nan"), "rmse": float("nan"), "mae": float("nan")}
    return {"bias": float(d.mean()), "rmse": float(np.sqrt((d**2).mean())), "mae": float(np.abs(d).mean())}
