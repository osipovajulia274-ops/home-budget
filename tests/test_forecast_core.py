import datetime as dt

import numpy as np

from app.services import forecast_core as fc


def _hist(n=120, value=1000.0):
    return np.full(n, value)


def test_features_shape_and_finite():
    x, scale, rem = fc.build_features(_hist(), dt.date(2026, 3, 10), salary_days=[5, 20])
    assert x.shape == (fc.WINDOW, fc.N_FEATURES) and x.dtype == np.float32
    assert np.isfinite(x).all()
    assert scale == 1000.0
    assert abs(rem - (31 - 10) / 31) < 1e-9


def test_dow_one_hot_and_dom_encoding():
    end = dt.date(2026, 3, 10)                       # Tuesday
    x, _, _ = fc.build_features(_hist(), end)
    assert x[-1, 1:8].tolist() == [0, 1, 0, 0, 0, 0, 0]
    assert np.allclose(x[-1, 8], np.sin(2 * np.pi * 10 / 31), atol=1e-6)
    assert np.allclose(x[-1, 9], np.cos(2 * np.pi * 10 / 31), atol=1e-6)
    assert (x[:, 1:8].sum(axis=1) == 1).all()


def test_payday_flag_only_when_known():
    end = dt.date(2026, 3, 10)
    known, _, _ = fc.build_features(_hist(), end, salary_days=[5])
    unknown, _, _ = fc.build_features(_hist(), end, salary_days=None)
    assert known[:, 10].sum() > 0 and unknown[:, 10].sum() == 0
    assert known[:, 10][-6] == 1.0                    # 10 March - 5 days = 5 March


def test_month_to_date_is_exact():
    h = np.zeros(fc.HISTORY_DAYS)
    h[-10:] = 100.0                                   # 10 days of 100 roubles ending 10 March
    x, scale, _ = fc.build_features(h, dt.date(2026, 3, 10))
    assert scale == 100.0
    assert np.isclose(x[-1, 11], np.log1p(1000.0 / 100.0))       # whole 10 days belong to March
    assert np.isclose(x[-10, 11], np.log1p(100.0 / 100.0))       # 1 March: only that day so far


def test_future_values_cannot_leak():
    h = np.random.default_rng(0).uniform(0, 3000, 200)
    a, _, _ = fc.build_features(h[:150], dt.date(2026, 3, 10))
    h2 = h.copy(); h2[150:] = 1e9
    b, _, _ = fc.build_features(h2[:150], dt.date(2026, 3, 10))
    assert np.array_equal(a, b)


def test_short_history_is_zero_padded():
    x, scale, _ = fc.build_features(np.full(30, 500.0), dt.date(2026, 3, 10))
    assert x.shape == (fc.WINDOW, fc.N_FEATURES)
    assert (x[:25, 0] == 0).all() and scale == 500.0


def test_decode_output_roundtrip_and_clip():
    assert np.isclose(fc.decode_output(np.log1p(2.5), 1000.0), 2500.0)
    assert fc.decode_output(-5.0, 1000.0) == 0.0


def test_baseline_avg():
    assert fc.baseline_avg(np.arange(100.0), 10) == np.arange(72.0, 100.0).mean() * 10
    assert fc.baseline_avg(np.array([]), 10) == 0.0


def test_baseline_dow_uses_weekday_profile():
    end = dt.date(2026, 3, 8)                          # Sunday
    days = [end - dt.timedelta(days=i) for i in range(55, -1, -1)]
    h = np.array([5000.0 if d.weekday() == 5 else 0.0 for d in days])   # spends only on Saturdays
    rest = fc.baseline_dow(h, end)
    saturdays = sum(1 for k in range(1, 24) if (end + dt.timedelta(days=k)).weekday() == 5)
    assert np.isclose(rest, 5000.0 * saturdays)
    assert fc.baseline_dow(h, dt.date(2026, 3, 31)) == 0.0          # nothing remains on the last day
    assert fc.baseline_dow(h, end, regular_rest=700.0) > rest


def _tx(month, day, sub, amount):
    return (dt.date(2026, month, day), sub, amount)


def test_regular_payments_detected_and_not_double_counted():
    hist = [_tx(m, 5, "housing.rent", 30000.0) for m in (1, 2, 3)]
    hist += [_tx(m, 20, "subscriptions.streaming", 399.0) for m in (1, 2, 3)]
    # 10 April: rent (day 5) has passed, the subscription (day 20) is still ahead
    hist_apr = hist + [_tx(4, 5, "housing.rent", 30000.0)]
    assert fc.expected_regular_rest(hist_apr, dt.date(2026, 4, 10)) == 399.0
    # 2 April: both are ahead
    assert fc.expected_regular_rest(hist, dt.date(2026, 4, 2)) == 30000.0 + 399.0
    # a payment seen in only one month is not regular
    assert fc.expected_regular_rest([_tx(3, 20, "subscriptions.other", 149.0)], dt.date(2026, 4, 2)) == 0.0
    # no history of the previous month: nothing expected
    assert fc.expected_regular_rest([], dt.date(2026, 4, 2)) == 0.0


def test_regular_sub_prefixes():
    assert fc.is_regular_sub("housing.rent") and fc.is_regular_sub("subscriptions.other")
    assert not fc.is_regular_sub("food.dairy")
