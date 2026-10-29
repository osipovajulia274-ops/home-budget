import datetime as dt

import pytest

from app.services import goals as g
from app.services import history as h

D = dt.date


def mk(months, days=(3, 10, 20), amount=1000.0):
    return [(D(y, m, d), amount) for y, m in months for d in days]


def test_monthly_totals_marks_complete_and_current():
    txs = mk([(2026, 7), (2026, 8), (2026, 9)]) + [(D(2026, 10, 2), 500.0)]
    t = h.monthly_totals(txs, D(2026, 10, 12))
    assert [x.month for x in t] == ["2026-07", "2026-08", "2026-09", "2026-10"]
    assert [x.complete for x in t] == [True, True, True, False]
    assert t[0].total == 3000.0 and t[-1].total == 500.0


def test_first_month_started_midway_is_not_complete():
    txs = [(D(2026, 8, 20), 100.0)] + mk([(2026, 9)])
    t = h.monthly_totals(txs, D(2026, 10, 5))
    assert [x.complete for x in t if x.month == "2026-08"] == [False]
    assert h.typical_month(t).n_months == 1


def test_months_before_first_transaction_are_not_listed():
    t = h.monthly_totals([(D(2026, 9, 5), 10.0)], D(2026, 10, 5), n_months=6)
    assert [x.month for x in t] == ["2026-09", "2026-10"]


def test_typical_month_none_without_complete_month_and_median():
    assert h.typical_month(h.monthly_totals([(D(2026, 10, 1), 5.0)], D(2026, 10, 5))) is None
    txs = [(D(2026, 7, 2), 1000.0), (D(2026, 8, 2), 5000.0), (D(2026, 9, 2), 2000.0)]
    tm = h.typical_month(h.monthly_totals(txs, D(2026, 10, 5)))
    assert (tm.median, tm.lowest, tm.highest, tm.n_months) == (2000.0, 1000.0, 5000.0, 3)


def test_pace_comparison_uses_same_day_of_month():
    txs = mk([(2026, 8), (2026, 9)]) + [(D(2026, 10, 2), 1500.0), (D(2026, 10, 9), 700.0)]
    p = h.pace_comparison(txs, D(2026, 10, 12))
    assert p.spent_now == 2200.0 and p.usual_by_now == 2000.0 and p.difference == 200.0
    assert p.share_of_usual == pytest.approx(1.1)


def test_pace_comparison_none_without_history_and_empty_input():
    assert h.pace_comparison([(D(2026, 10, 1), 5.0)], D(2026, 10, 5)) is None
    assert h.pace_comparison([], D(2026, 10, 5)) is None
    assert h.monthly_totals([], D(2026, 10, 5)) == []


def test_pace_comparison_short_month_cutoff():
    # today = 31 Mar; February has 28 days -> the whole of February counts
    txs = [(D(2026, 2, 1), 50.0), (D(2026, 2, 28), 50.0), (D(2026, 3, 31), 10.0)]
    p = h.pace_comparison(txs, D(2026, 3, 31))
    assert p.usual_by_now == 100.0 and p.spent_now == 10.0


def test_goal_requires_positive_limit():
    with pytest.raises(ValueError):
        g.Goal("total", 0)


def test_scope_matching_category_vs_subcategory():
    s = {"food.dairy": 100.0, "food.meat": 50.0, "foodcourt.x": 7.0, "eating_out.cafe": 30.0}
    assert g.spent_for_scope("total", s) == 187.0
    assert g.spent_for_scope("food", s) == 150.0           # not "foodcourt.x"
    assert g.spent_for_scope("food.dairy", s) == 100.0


@pytest.mark.parametrize("spent,day,level", [
    (100, 15, "ok"), (700, 10, "ahead_of_pace"), (950, 10, "watch"), (1000, 10, "exceeded"), (1200, 30, "exceeded"),
    (150, 2, "ok"),                                         # early-month noise: 15% used, below 20%
])
def test_levels(spent, day, level):
    assert g.evaluate_goal(g.Goal("total", 1000), spent, D(2026, 9, day)).level == level


def test_status_numbers():
    st = g.evaluate_goal(g.Goal("total", 3000), 2200, D(2026, 10, 12))
    assert st.remaining == 800 and st.over_by == 0 and st.days_left == 19
    assert st.per_day_left == pytest.approx(42.11, abs=0.01)
    over = g.evaluate_goal(g.Goal("total", 1000), 1300, D(2026, 10, 12))
    assert over.remaining == 0 and over.over_by == 300


def test_last_day_does_not_divide_by_zero():
    st = g.evaluate_goal(g.Goal("total", 1000), 900, D(2026, 9, 30))
    assert st.days_left == 0 and st.per_day_left == 100 and st.level == "watch"


def test_recommendations_texts_and_silence_when_ok():
    titles = {"food": "Продукты"}
    sts = g.evaluate_goals([g.Goal("total", 100000), g.Goal("food", 1000), g.Goal("eating_out", 1000)],
                           {"food.dairy": 1200.0, "eating_out.cafe": 100.0}, D(2026, 10, 12))
    recs = g.goal_recommendations(sts, titles)
    assert len(recs) == 1                                  # only the exceeded one speaks
    r = recs[0]
    assert r["rule_id"] == "R02_goal_exceeded" and r["severity"] == "alert" and r["category"] == "food"
    assert "Продукты" in r["title"] and "200" in r["title"]


def test_no_forecast_wording_in_texts():
    sts = g.evaluate_goals([g.Goal("total", 3000)], {"food.dairy": 2200.0}, D(2026, 10, 12))
    for r in g.goal_recommendations(sts):
        text = (r["title"] + r["body"]).lower()
        assert "прогноз" not in text and "до конца месяца потрат" not in text


def test_fmt_rub():
    assert h.fmt_rub(12345.6) == "12 346 ₽"
