from app.services import analytics as an
from tests.helpers import D, TODAY, base_history, mk


def test_parse_month():
    assert an.parse_month("2026-10") == D(2026, 10, 1)
    for bad in ("2026-13", "26-10", "", "2026-1"):
        try:
            an.parse_month(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_summary_shares_deltas_and_transfers_excluded():
    txs = [mk(D(2026, 9, 5), "food.dairy", 1000), mk(D(2026, 9, 6), "eating_out.cafe", 1000),
           mk(D(2026, 10, 5), "food.dairy", 1500), mk(D(2026, 10, 6), "food.produce", 500),
           mk(D(2026, 10, 7), "eating_out.cafe", 2500), mk(D(2026, 10, 8), "other.transfers", 99999)]
    s = an.summary(an.spending(txs), D(2026, 10, 1), TODAY)
    assert s["total"] == 4500.0 and s["prev_month_total"] == 2000.0 and s["delta_vs_prev"] == 1.25
    cats = {c["category"]: c for c in s["by_category"]}
    assert "other" not in cats
    assert cats["food"]["amount"] == 2000.0 and cats["food"]["prev_amount"] == 1000.0 and cats["food"]["delta_vs_prev"] == 1.0
    assert abs(sum(c["share"] for c in s["by_category"]) - 1.0) < 1e-3          # shares add up to 100%
    assert [c["category"] for c in s["by_category"]] == ["eating_out", "food"]  # biggest first
    assert s["top_items"][0]["amount"] == 2500.0 and s["days_with_data"] == 3


def test_summary_without_previous_month_has_no_delta():
    s = an.summary(an.spending([mk(D(2026, 10, 5), "food.dairy", 100)]), D(2026, 10, 1), TODAY)
    assert s["prev_month_total"] is None and s["delta_vs_prev"] is None
    assert s["by_category"][0]["delta_vs_prev"] is None


def test_summary_of_empty_month():
    s = an.summary([], D(2026, 10, 1), TODAY)
    assert s["total"] == 0 and s["by_category"] == [] and s["goal_total"] is None


def test_summary_includes_goal_status():
    from app.services.goals import Goal, evaluate_goals
    spend = an.spending([mk(D(2026, 10, 5), "food.dairy", 950)])
    st = evaluate_goals([Goal("food", 1000), Goal("total", 5000)], an.spend_by_sub(spend), TODAY)
    s = an.summary(spend, D(2026, 10, 1), TODAY, st)
    assert s["by_category"][0]["goal_limit"] == 1000 and s["by_category"][0]["goal_used"] == 0.95
    assert s["goal_total"]["limit"] == 5000


def test_detect_recurring_needs_28_31_days_and_same_amount():
    ok = [mk(D(2026, 8, 5), "subscriptions.streaming", 399, "Кино подписка"),
          mk(D(2026, 9, 5), "subscriptions.streaming", 399, "Кино подписка"),
          mk(D(2026, 10, 5), "subscriptions.streaming", 405, "Кино подписка")]        # +1.5% is within 5%
    r = an.detect_recurring(ok, TODAY)
    assert len(r) == 1 and r[0].occurrences == 3 and r[0].amount == 405 and r[0].interval_days in (30, 31)
    assert an.detect_recurring([ok[0], mk(D(2026, 9, 20), "subscriptions.streaming", 399, "Кино подписка")], TODAY) == []
    assert an.detect_recurring([ok[0], mk(D(2026, 9, 5), "subscriptions.streaming", 600, "Кино подписка")], TODAY) == []
    assert an.detect_recurring([ok[0]], TODAY) == []


def test_old_subscription_is_dropped():
    old = [mk(D(2026, 3, 5), "subscriptions.other", 100, "Старая"), mk(D(2026, 4, 5), "subscriptions.other", 100, "Старая")]
    assert an.detect_recurring(old, TODAY) == []


def test_recurring_uses_only_the_latest_unbroken_run():
    t = "Облако"
    rows = [mk(D(2026, 4, 5), "subscriptions.other", 100, t), mk(D(2026, 5, 5), "subscriptions.other", 100, t),
            mk(D(2026, 7, 5), "subscriptions.other", 100, t), mk(D(2026, 8, 5), "subscriptions.other", 100, t),
            mk(D(2026, 9, 5), "subscriptions.other", 100, t)]          # June was skipped: a new run starts in July
    r = an.detect_recurring(rows, TODAY)
    assert len(r) == 1 and r[0].occurrences == 3 and r[0].last_date == D(2026, 9, 5)


def test_recurring_ignores_transfers():
    rows = [mk(D(2026, 9, 5), "other.transfers", 5000, "Перевод"), mk(D(2026, 10, 5), "other.transfers", 5000, "Перевод")]
    assert an.detect_recurring(rows, TODAY) == []
