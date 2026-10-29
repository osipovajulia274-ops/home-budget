"""Every rule: one case where it fires and one where it must not (guide section 9)."""
import datetime as dt

from app.services import rules
from app.services.goals import Goal, evaluate_goals
from app.services import analytics as an
from tests.helpers import D, TODAY, base_history, mk


def ctx(txs, today=TODAY, income=None, goals=()):
    spend = an.spending(txs)
    first = today.replace(day=1)
    st = evaluate_goals(goals, an.spend_by_sub(an.in_month(spend, first)), today) if goals else []
    return rules.UserCtx(today=today, month=first, txs=spend, goal_statuses=st, income=income)


def nb(s):
    """Amounts use non-breaking spaces; compare with ordinary ones."""
    return s.replace("\xa0", " ")


def ids(cards):
    return [c["rule_id"] for c in cards]


def fire(rule, c):
    return rule(c)


# ------------------------------------------------------------------------------ goals ----
def test_goal_cards_fire_and_stay_quiet():
    over = ctx(base_history() + [mk(D(2026, 10, 5), "food.dairy", 6000)], goals=[Goal("food", 5000)])
    assert ids(rules.r_goals(over)) == ["R02_goal_exceeded"]
    quiet = ctx(base_history() + [mk(D(2026, 10, 5), "food.dairy", 500)], goals=[Goal("food", 5000)])
    assert rules.r_goals(quiet) == []
    c = rules.r_goals(over)[0]
    assert "прогноз" not in (c["title"] + c["body"]).lower() and c["payload"]["key"].startswith("R02_goal_exceeded")


# --------------------------------------------------------------------------------- R03 ----
def _past(sub, amount, day=10):
    return [mk(D(2026, m, day), sub, amount) for m in (7, 8, 9)]


def test_r03_fires_when_month_so_far_is_30_percent_above_usual():
    c = ctx(base_history() + _past("eating_out.delivery", 1000) + [mk(D(2026, 10, 10), "eating_out.delivery", 3000)])
    cards = rules.r03_category_growth(c)
    assert ids(cards) == ["R03_category_growth"] and cards[0]["category"] == "eating_out"
    assert "200%" in cards[0]["title"] and cards[0]["severity"] == "warning"


def test_r03_quiet_for_small_difference_or_small_amount():
    assert rules.r03_category_growth(ctx(base_history() + _past("eating_out.delivery", 1000)
                                         + [mk(D(2026, 10, 10), "eating_out.delivery", 1200)])) == []
    assert rules.r03_category_growth(ctx(base_history() + _past("eating_out.coffee", 100)
                                         + [mk(D(2026, 10, 10), "eating_out.coffee", 600)])) == []   # +500 < 1000


def test_r03_needs_a_complete_month_to_compare():
    assert rules.r03_category_growth(ctx([mk(D(2026, 10, 5), "eating_out.cafe", 9000)])) == []


# --------------------------------------------------------------------------------- R04 ----
def test_r04_fires_above_20_percent_and_above_own_median():
    hist = [mk(D(2026, m, 5), "food.dairy", 900) for m in (7, 8, 9)] + [mk(D(2026, m, 6), "food.sweets_snacks", 100) for m in (7, 8, 9)]
    cur = [mk(D(2026, 10, 5), "food.dairy", 750), mk(D(2026, 10, 6), "food.sweets_snacks", 250)]
    cards = rules.r04_sweets_share(ctx(base_history() + hist + cur))
    assert ids(cards) == ["R04_sweets_share"] and "25%" in cards[0]["title"]


def test_r04_quiet_below_20_percent_or_not_above_usual():
    low = [mk(D(2026, 10, 5), "food.dairy", 900), mk(D(2026, 10, 6), "food.sweets_snacks", 100)]
    assert rules.r04_sweets_share(ctx(base_history() + low)) == []
    hist = [mk(D(2026, m, 5), "food.dairy", 700) for m in (7, 8, 9)] + [mk(D(2026, m, 6), "food.sweets_snacks", 300) for m in (7, 8, 9)]
    cur = [mk(D(2026, 10, 5), "food.dairy", 750), mk(D(2026, 10, 6), "food.sweets_snacks", 250)]   # 25% but usual is 30%
    assert rules.r04_sweets_share(ctx(base_history() + hist + cur)) == []


# --------------------------------------------------------------------------------- R05 ----
def _milk(prices):
    return [mk(D(2026, 7 + i, 5), "food.dairy", p, "Молоко 3,2% 900мл", qty=1) for i, p in enumerate(prices)]


def test_r05_fires_for_a_price_rise_of_15_percent():
    cards = rules.r05_price_increase(ctx(base_history() + _milk([80, 90, 100])))
    assert ids(cards) == ["R05_price_increase"] and "25%" in cards[0]["title"]


def test_r05_quiet_for_small_rise_or_few_purchases():
    assert rules.r05_price_increase(ctx(base_history() + _milk([80, 81, 82]))) == []
    assert rules.r05_price_increase(ctx(base_history() + _milk([80, 100]))) == []


def test_r05_uses_price_per_unit():
    rows = [mk(D(2026, 7, 5), "food.produce", 100, "Бананы", qty=1.0), mk(D(2026, 8, 5), "food.produce", 100, "Бананы", qty=1.0),
            mk(D(2026, 9, 5), "food.produce", 200, "Бананы", qty=2.0)]            # 2 kg for 200 = same unit price
    assert rules.r05_price_increase(ctx(base_history() + rows)) == []


# --------------------------------------------------------------------------------- R06 ----
def test_r06_fires_for_many_small_trips_and_not_for_few_or_large():
    many = [mk(D(2026, 10, 1 + i), "food.grocery", 300, receipt=i) for i in range(13)]
    assert ids(rules.r06_small_trips(ctx(many))) == ["R06_small_frequent_trips"]
    assert rules.r06_small_trips(ctx(many[:12])) == []
    big = [mk(D(2026, 10, 1 + i), "food.grocery", 900, receipt=i) for i in range(13)]
    assert rules.r06_small_trips(ctx(big)) == []


def test_r06_items_of_one_receipt_are_one_trip():
    rows = [mk(D(2026, 10, 1 + i % 28), "food.grocery", 40, receipt=i // 4) for i in range(60)]      # 15 receipts x 4 items
    assert rules.r06_small_trips(ctx(rows)) != []


# --------------------------------------------------------------------------------- R07 ----
def test_r07_coffee_threshold_is_eight():
    coffee = lambda n: [mk(D(2026, 10, 1 + i), "eating_out.coffee", 250) for i in range(n)]       # noqa: E731
    assert ids(rules.r07_coffee(ctx(coffee(8)))) == ["R07_coffee_to_go"]
    assert rules.r07_coffee(ctx(coffee(7))) == []


# --------------------------------------------------------------------------------- R08 ----
def test_r08_delivery_share():
    f = mk(D(2026, 10, 3), "food.grocery", 10000)
    assert ids(rules.r08_delivery_vs_home(ctx([f, mk(D(2026, 10, 4), "eating_out.delivery", 2000)]))) == ["R08_delivery_vs_home"]
    assert rules.r08_delivery_vs_home(ctx([f, mk(D(2026, 10, 4), "eating_out.delivery", 1500)])) == []   # exactly 15%
    assert rules.r08_delivery_vs_home(ctx([mk(D(2026, 10, 4), "eating_out.delivery", 2000)])) == []      # no food at all


# --------------------------------------------------------------------------------- R09 ----
def test_r09_taxi_vs_public():
    assert ids(rules.r09_taxi_vs_public(ctx([mk(D(2026, 10, 3), "transport.taxi", 5400), mk(D(2026, 10, 4), "transport.public", 500)]))) == ["R09_taxi_vs_public"]
    assert rules.r09_taxi_vs_public(ctx([mk(D(2026, 10, 3), "transport.taxi", 5400), mk(D(2026, 10, 4), "transport.public", 2000)])) == []
    assert rules.r09_taxi_vs_public(ctx([mk(D(2026, 10, 3), "transport.taxi", 2500)])) == []


# -------------------------------------------------------------------------------- R10 ----
def _sub(title, amount, months=(8, 9, 10)):
    return [mk(D(2026, m, 5), "subscriptions.streaming", amount, title) for m in months]


def test_r10_two_subscriptions_fire_with_month_and_year_sums():
    cards = rules.r10_subscriptions(ctx(_sub("Кино", 399) + _sub("Музыка", 299)))
    assert ids(cards) == ["R10_subscriptions_found"]
    t = nb(cards[0]["title"])
    assert "2 подписки" in t and "698 ₽ в месяц" in t and "8 376 ₽ в год" in t


def test_r10_one_subscription_is_not_enough():
    assert rules.r10_subscriptions(ctx(_sub("Кино", 399))) == []


# -------------------------------------------------------------------------------- R11 ----
def test_r11_recurring_gym_charge_only():
    gym = [mk(D(2026, m, 1), "health.sport", 3200, "Абонемент фитнес") for m in (9, 10)]
    cards = rules.r11_gym_charge(ctx(gym))
    assert ids(cards) == ["R11_unused_gym"] and "не видит" in cards[0]["body"]       # honest about what the app knows
    assert rules.r11_gym_charge(ctx(gym[:1])) == []


# -------------------------------------------------------------------------------- R12 ----
def test_r12_pharmacy():
    assert ids(rules.r12_pharmacy_growth(ctx(base_history() + _past("health.pharmacy", 500) + [mk(D(2026, 10, 10), "health.pharmacy", 1500)]))) == ["R12_pharmacy_growth"]
    assert rules.r12_pharmacy_growth(ctx(base_history() + _past("health.pharmacy", 500) + [mk(D(2026, 10, 10), "health.pharmacy", 600)])) == []


# -------------------------------------------------------------------------------- R13 ----
def _last_year(peak_extra):
    rows = [mk(D(2025, m, 10), "family_kids.goods", 2000) for m in (6, 7, 8)]
    rows.append(mk(D(2025, 9, 10), "family_kids.goods", 2000 + peak_extra))
    return rows


def test_r13_seasonal_peak_within_45_days():
    c = ctx(_last_year(9000) + [mk(D(2026, 8, 5), "food.dairy", 100)], today=D(2026, 8, 20))
    cards = rules.r13_seasonal(c)
    assert ids(cards) == ["R13_seasonal_prep"]
    body = nb(cards[0]["body"])
    assert "9 000 ₽" in body and "До сентября осталось 12 дн." in body and "в сентябре" in body


def test_r13_quiet_when_peak_far_small_or_unknown():
    assert rules.r13_seasonal(ctx(_last_year(9000), today=D(2026, 6, 1))) == []            # > 45 days away
    assert rules.r13_seasonal(ctx(_last_year(1000), today=D(2026, 8, 20))) == []            # small rise
    assert rules.r13_seasonal(ctx([mk(D(2026, 8, 5), "food.dairy", 100)], today=D(2026, 8, 20))) == []   # no history


# -------------------------------------------------------------------------------- R14 ----
def _normal_days(n=20):
    return [mk(D(2026, 9, 1 + i), "food.grocery", 400 + 20 * (i % 6)) for i in range(n)]


def test_r14_anomalous_day():
    cards = rules.r14_anomaly_day(ctx(_normal_days() + [mk(D(2026, 10, 27), "home.appliances", 18200)]))
    assert ids(cards) == ["R14_anomaly_day"] and "27 октября" in cards[0]["title"] and "18 200 ₽" in nb(cards[0]["title"])


def test_r14_quiet_for_modest_days_few_days_and_rent():
    assert rules.r14_anomaly_day(ctx(_normal_days() + [mk(D(2026, 10, 27), "food.grocery", 2000)])) == []   # < 3000
    assert rules.r14_anomaly_day(ctx(_normal_days(10) + [mk(D(2026, 10, 27), "home.appliances", 18200)])) == []   # not enough days
    assert rules.r14_anomaly_day(ctx(_normal_days() + [mk(D(2026, 10, 2), "housing.rent", 38000)])) == []   # rent is regular


# -------------------------------------------------------------------------------- R15 ----
def test_r15_unknown_items():
    unk = lambda n: [mk(D(2026, 10, 1 + i), "other.unknown", 100) for i in range(n)]              # noqa: E731
    assert ids(rules.r15_unknown(ctx(unk(5)))) == ["R15_unknown_items"]
    assert rules.r15_unknown(ctx(unk(4))) == []


# -------------------------------------------------------------------------------- R16 ----
def test_r16_reference_needs_income_and_a_complete_month():
    hist = base_history() + [mk(D(2026, 9, 10), "food.grocery", 8000)]
    cards = rules.r16_reference_shares(ctx(hist, income=60000))
    assert ids(cards) == ["R16_reference_shares"] and "Росстат" in cards[0]["body"]
    assert rules.r16_reference_shares(ctx(hist, income=None)) == []
    assert rules.r16_reference_shares(ctx([mk(D(2026, 10, 3), "food.grocery", 100)], income=60000)) == []


# --------------------------------------------------------------------------- engine ----
def test_run_rules_orders_by_severity_then_effect_and_caps_at_five():
    txs = (base_history() + _past("eating_out.delivery", 1000) + [mk(D(2026, 10, 10), "eating_out.delivery", 3000)]
           + [mk(D(2026, 10, 1 + i), "eating_out.coffee", 250) for i in range(9)]
           + [mk(D(2026, 10, 3), "transport.taxi", 5400)] + [mk(D(2026, 10, 4), "food.grocery", 400)])
    c = ctx(txs, goals=[Goal("total", 5000)])
    cards = rules.run_rules(c)
    sev = [rules.SEVERITY_ORDER[x["severity"]] for x in cards]
    assert sev == sorted(sev) and cards[0]["rule_id"] == "R02_goal_exceeded"
    assert len(rules.pick_active(cards)) <= 5
    hidden = [cards[0]["payload"]["key"]]
    assert cards[0] not in rules.pick_active(cards, hidden)


def test_texts_are_calm_and_free_of_advice_words():
    from tests.test_demo import demo_cards
    for c in demo_cards():
        text = (c["title"] + " " + c["body"] + " " + (c["action"] or "")).lower()
        for word in ("инвест", "кредит", "акци", "облигац", "вложи", "купите"):
            assert word not in text, (c["rule_id"], word)


def test_plural():
    assert [rules.plural(n, "подписка", "подписки", "подписок") for n in (1, 2, 5, 11, 21, 112)] == \
           ["подписка", "подписки", "подписок", "подписок", "подписка", "подписок"]
