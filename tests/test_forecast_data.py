import numpy as np
import pandas as pd
import pytest

from app.services.forecast_core import N_FEATURES, WINDOW
from ml import forecast_data as fd
from ml.transactions import generate_history


@pytest.fixture(scope="module")
def prepared():
    tx, users = generate_history(n_users=12, seed=11)
    series = fd.build_series(tx, users, salary_known_frac=0.5, seed=1)
    return tx, users, series


def test_sample_points_rules(prepared):
    s = next(iter(prepared[2].values()))
    pts = fd.sample_points(s, fd.EVAL_DAYS)
    assert pts, "no sample points"
    for i, m_end in pts:
        d = s.date_at(i)
        assert i >= WINDOW - 1                             # a full window of history exists
        assert s.date_at(m_end).month == d.month           # target ends on the last day of the same month
        assert (s.date_at(m_end + 1).day == 1)
        assert m_end < s.n_days                            # the target month is fully inside the data


def test_samples_shapes_and_targets(prepared):
    S = fd.build_samples(prepared[2], fd.EVAL_DAYS)
    n = len(S.y)
    assert S.x.shape == (n, WINDOW, N_FEATURES) and S.rem.shape == (n, 1)
    assert np.isfinite(S.x).all() and np.isfinite(S.y).all()
    assert np.allclose(np.expm1(S.y) * S.scale, S.rest, rtol=1e-4, atol=1e-2)
    assert (S.spent >= 0).all() and (S.rest >= 0).all()
    assert set(S.day.tolist()) <= set(fd.EVAL_DAYS)


def test_target_does_not_leak_into_features(prepared):
    """Zeroing everything after day d of a month must not change the features of the sample at d."""
    uid, s = next(iter(prepared[2].items()))
    i, m_end = fd.sample_points(s, [10])[3]
    full = fd.build_samples({uid: s}, [10])
    k = [j for j, (ii, _) in enumerate(fd.sample_points(s, [10])) if ii == i][0]
    s2 = fd.UserSeries(s.user_id, s.start, s.daily.copy(), s.regular_daily.copy(), list(s.reg_tx), s.salary_days)
    s2.daily[i + 1:] = 0.0
    s2.regular_daily[i + 1:] = 0.0
    changed = fd.build_samples({uid: s2}, [10])
    assert np.array_equal(full.x[k], changed.x[k])
    assert np.isclose(full.b1[k], changed.b1[k]) and np.isclose(full.spent[k], changed.spent[k])


def test_baselines_are_sane(prepared):
    S = fd.build_samples(prepared[2], fd.EVAL_DAYS)
    res = fd.evaluate_methods(S, {"baseline_avg": S.b1, "baseline_dow": S.b2})
    for name in res:
        assert 0 < res[name]["overall"]["mape"] < 1.0, name
        assert set(res[name]["by_day"]) == {str(d) for d in fd.EVAL_DAYS}
    # knowing weekday profile and regular payments should not be worse than a flat average
    assert res["baseline_dow"]["overall"]["mae"] < res["baseline_avg"]["overall"]["mae"]


def test_choose_method_requires_5_percent():
    mk = lambda m: {"overall": {"mae": m, "mape": 0.1, "n": 1}}
    assert fd.choose_method({"baseline_avg": mk(120), "baseline_dow": mk(100), "lstm": mk(96)})[0] == "baseline_dow"
    assert fd.choose_method({"baseline_avg": mk(120), "baseline_dow": mk(100), "lstm": mk(94)})[0] == "lstm"


def test_interval_quantiles_order(prepared):
    S = fd.build_samples(prepared[2], fd.EVAL_DAYS)
    q = fd.interval_quantiles(S, S.b2)
    assert 0 < q["q_lo"] < 1.0 < q["q_hi"] * 1.5
