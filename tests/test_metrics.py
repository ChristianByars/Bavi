# Unit tests for metrics.py
import math
import warnings

import numpy as np
import pytest

import metrics

LAT = np.array([0.0, 60.0])


@pytest.fixture(autouse=True)
def _warnings_are_errors():
    """Any numpy/Python warning raised inside a metric fails the test, whichever way pytest is invoked."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def _brute_wmean(values, lat):
    """cos(latitude)-weighted mean of a (lat, lon) nested list, skipping NaN, with explicit loops."""
    num = den = 0.0
    for i, row in enumerate(values):
        w = math.cos(math.radians(lat[i]))
        for v in row:
            if not math.isnan(v):
                num += w * v
                den += w
    return num / den


def _brute_pattern_corr(fc, ref, lat):
    """Weighted centred correlation over the jointly valid points, with explicit loops."""
    pts = [
        (math.cos(math.radians(lat[i])), fc[i][j], ref[i][j])
        for i in range(len(fc))
        for j in range(len(fc[0]))
        if not (math.isnan(fc[i][j]) or math.isnan(ref[i][j]))
    ]
    sw = sum(w for w, _, _ in pts)
    mf = sum(w * f for w, f, _ in pts) / sw
    mr = sum(w * r for w, _, r in pts) / sw
    cov = sum(w * (f - mf) * (r - mr) for w, f, r in pts) / sw
    vf = sum(w * (f - mf) ** 2 for w, f, _ in pts) / sw
    vr = sum(w * (r - mr) ** 2 for w, _, r in pts) / sw
    return cov / math.sqrt(vf * vr)


def test_lat_weights_follow_cosine_and_have_mean_one():
    w = metrics.lat_weights(LAT)
    assert w.mean() == pytest.approx(1.0)
    assert w[0] / w[1] == pytest.approx(2.0)


def test_weighted_bias_uses_area_weights():
    fc = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    assert metrics.weighted_bias(fc, np.zeros_like(fc), LAT) == pytest.approx(2.0 / 3.0)


def test_constant_offset_gives_matching_rmse_and_mae():
    ref = np.random.default_rng(1).normal(size=(2, 4))
    assert metrics.weighted_rmse(ref + 2.0, ref, LAT) == pytest.approx(2.0)
    assert metrics.weighted_mae(ref - 3.0, ref, LAT) == pytest.approx(3.0)


def test_pattern_corr_is_one_minus_one_and_shift_scale_invariant():
    lat = np.linspace(-40, 40, 5)
    ref = np.random.default_rng(0).normal(size=(5, 8))
    assert metrics.pattern_corr(ref, ref, lat) == pytest.approx(1.0)
    assert metrics.pattern_corr(-ref, ref, lat) == pytest.approx(-1.0)
    assert metrics.pattern_corr(3.0 * ref + 7.0, ref, lat) == pytest.approx(1.0)


def test_nans_are_ignored():
    ref = np.ones((2, 3))
    ref[0, 0] = np.nan
    fc = np.ones((2, 3)) + 1.0
    assert metrics.weighted_bias(fc, ref, LAT) == pytest.approx(1.0)
    assert metrics.weighted_rmse(fc, ref, LAT) == pytest.approx(1.0)


def test_leading_dimensions_are_preserved():
    ref = np.zeros((3, 2, 4))
    out = metrics.weighted_bias(ref + np.array([1.0, 2.0, 3.0])[:, None, None], ref, LAT)
    assert out.shape == (3,) and out == pytest.approx([1.0, 2.0, 3.0])


def test_skill_row_keys_and_values():
    ref = np.random.default_rng(2).normal(size=(5, 8))
    row = metrics.skill_row(ref + 1.0, ref, np.linspace(-40, 40, 5))
    assert list(row) == ["Bias", "RMSE", "MAE", "Pattern corr"]
    assert row["Bias"] == 1.0 and row["Pattern corr"] == 1.0


def test_event_total_error_reports_percent_bias():
    out = metrics.event_total_error(np.full((2, 3), 12.0), np.full((2, 3), 10.0), LAT)
    assert out == {"Forecast area-mean": 12.0, "Reference area-mean": 10.0, "Total bias (%)": 20.0}


def test_haversine_known_distances():
    assert metrics.haversine_km(0, 0, 0, 90) == pytest.approx(np.pi / 2 * metrics.R_EARTH_KM, rel=1e-9)
    assert metrics.haversine_km(10, 20, 10, 20) == 0.0
    assert metrics.haversine_km(0, 179, 0, -179) == pytest.approx(2 * np.pi * metrics.R_EARTH_KM / 180, rel=1e-6)
    assert metrics.haversine_km(np.array([0.0, 1.0]), 0, 0, 0).shape == (2,)


def test_intensity_error():
    out = metrics.intensity_error([1000.0, 990.0, np.nan], [998.0, 992.0, 1000.0])
    assert out == {"bias": 0.0, "rmse": 2.0, "mae": 2.0}
    assert np.isnan(metrics.intensity_error([np.nan], [1.0])["rmse"])


# A non-trivial 3-latitude pair in which the area weighting changes the pattern correlation.
LAT3 = np.array([0.0, 45.0, 80.0])
REF3 = np.array([[-0.7, -0.2, 1.7, 0.7], [-0.1, 1.5, 0.9, 1.6], [-3.6, -1.8, -1.8, -0.4]])
FC3 = np.array([[-0.1, 0.4, -0.3, 2.4], [-1.8, 1.9, 0.2, 0.2], [-2.8, -1.7, -0.7, -0.4]])


def test_weighted_rmse_weights_a_non_constant_difference():
    fc = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
    ref = np.zeros_like(fc)
    # weights (4/3, 2/3): mean square = (4/3 * 1 + 2/3 * 9) / 2 = 11/3; the unweighted answer is sqrt(5)
    got = metrics.weighted_rmse(fc, ref, LAT)
    assert got == pytest.approx(math.sqrt(11.0 / 3.0))
    assert got == pytest.approx(math.sqrt(_brute_wmean(((fc - ref) ** 2).tolist(), LAT)))
    assert abs(got - math.sqrt(5.0)) > 0.3


def test_weighted_mae_weights_a_non_constant_difference():
    fc = np.array([[1.0, -1.0, 1.0], [3.0, -3.0, 3.0]])
    ref = np.zeros_like(fc)
    # |difference| is 1 on the equator row and 3 on the 60 degree row: (4/3 * 1 + 2/3 * 3) / 2 = 5/3, unweighted 2
    got = metrics.weighted_mae(fc, ref, LAT)
    assert got == pytest.approx(5.0 / 3.0)
    assert got == pytest.approx(_brute_wmean(np.abs(fc - ref).tolist(), LAT))


def test_weighted_mae_ignores_nan_in_one_field():
    fc = np.array([[np.nan, 2.0, 3.0], [4.0, 5.0, 6.0]])
    ref = np.ones((2, 3))
    # valid |differences| 1, 2 (equator row) and 3, 4, 5 (60 degree row): (4/3 * 3 + 2/3 * 12) / (4/3 * 2 + 2/3 * 3) = 18/7
    got = metrics.weighted_mae(fc, ref, LAT)
    assert got == pytest.approx(18.0 / 7.0)
    assert got == pytest.approx(_brute_wmean(np.abs(fc - ref).tolist(), LAT))
    assert abs(got - 3.0) > 0.2  # the unweighted nanmean of the same differences


def test_pattern_corr_is_cos_latitude_weighted():
    got = metrics.pattern_corr(FC3, REF3, LAT3)
    assert got == pytest.approx(_brute_pattern_corr(FC3.tolist(), REF3.tolist(), LAT3))
    assert got == pytest.approx(0.48786, abs=1e-5)
    assert abs(got - np.corrcoef(FC3.ravel(), REF3.ravel())[0, 1]) > 0.1  # unweighted correlation is about 0.718


def test_pattern_corr_uses_only_jointly_valid_points_when_nans_differ():
    fc, ref = FC3.copy(), REF3.copy()
    fc[0, 1] = np.nan
    fc[2, 3] = np.nan
    ref[1, 2] = np.nan
    ref[2, 0] = np.nan
    got = metrics.pattern_corr(fc, ref, LAT3)
    assert np.isfinite(got)
    assert got == pytest.approx(_brute_pattern_corr(fc.tolist(), ref.tolist(), LAT3))
    ok = np.isfinite(fc) & np.isfinite(ref)
    assert abs(got - np.corrcoef(fc[ok], ref[ok])[0, 1]) > 0.1  # unweighted correlation of the same points


def test_event_total_error_with_zero_reference_is_nan_not_an_error():
    out = metrics.event_total_error(np.ones((2, 3)), np.zeros((2, 3)), LAT)
    assert out["Forecast area-mean"] == 1.0 and out["Reference area-mean"] == 0.0
    assert math.isnan(out["Total bias (%)"])


@pytest.mark.parametrize("c", [0.0, 1.0, 0.1, 273.15, 3.3, 1e5 / 3])
def test_pattern_corr_is_nan_for_constant_fields(c):
    lat = np.linspace(-40, 40, 5)
    const = np.full((5, 8), c)
    noisy = np.random.default_rng(3).normal(size=(5, 8))
    assert np.isnan(metrics.pattern_corr(const, const, lat))
    assert np.isnan(metrics.pattern_corr(const, noisy, lat))
    assert np.isnan(metrics.pattern_corr(noisy, const, lat))
    with_hole = const.copy()
    with_hole[2, 3] = np.nan
    assert np.isnan(metrics.pattern_corr(with_hole, noisy, lat))


def test_pattern_corr_nan_only_for_the_degenerate_member_of_a_stack():
    lat = np.linspace(-40, 40, 5)
    noisy = np.random.default_rng(3).normal(size=(5, 8))
    stack = np.stack([np.full((5, 8), 273.15), noisy])
    out = metrics.pattern_corr(stack, np.stack([noisy, noisy]), lat)
    assert out.shape == (2,) and np.isnan(out[0]) and out[1] == pytest.approx(1.0)


def test_pattern_corr_normal_cases_are_unchanged():
    ref = np.array([[1.0, 2.0], [3.0, 4.0]])
    fc = np.array([[1.0, 2.0], [3.0, 5.0]])
    # equal weights (both rows on the equator): Pearson r = 6.5 / sqrt(8.75 * 5)
    assert metrics.pattern_corr(fc, ref, [0.0, 0.0]) == pytest.approx(6.5 / math.sqrt(43.75))
    # a genuine small anomaly on a large offset is still a valid pattern, not a "constant"
    lat = np.linspace(-40, 40, 5)
    t = 273.15 + 1e-3 * np.random.default_rng(4).normal(size=(5, 8))
    assert metrics.pattern_corr(t, t, lat) == pytest.approx(1.0)
    assert metrics.pattern_corr(-t, t, lat) == pytest.approx(-1.0)


def test_all_nan_inputs_give_nan_quietly():
    nan = np.full((2, 3), np.nan)
    ones = np.ones((2, 3))
    row = metrics.skill_row(nan, nan, LAT)
    assert list(row) == ["Bias", "RMSE", "MAE", "Pattern corr"]
    assert all(math.isnan(v) for v in row.values())
    for fn in (metrics.weighted_bias, metrics.weighted_rmse, metrics.weighted_mae, metrics.pattern_corr):
        assert np.isnan(fn(nan, nan, LAT)) and np.isnan(fn(nan, ones, LAT)) and np.isnan(fn(ones, nan, LAT))
