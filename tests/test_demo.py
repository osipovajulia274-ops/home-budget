import datetime as dt

from app.services import analytics as an
from app.services import demo, rules
from app.services.goals import Goal, evaluate_goals
from app.services.normalize import normalize
from app.services import taxonomy as tax
from tests.helpers import TODAY, D


def demo_txs(today=TODAY):
    return [an.Tx(id=i, date=x["date"], title=x["title"], title_norm=normalize(x["title"]), amount=x["amount"],
                  subcategory=x["subcategory"], qty=x["qty"]) for i, x in enumerate(demo.flat_items(demo.generate(today)))]


def demo_cards(income=120000):
    spend = an.spending(demo_txs())
    first = TODAY.replace(day=1)
    st = evaluate_goals([Goal("total", 75000), Goal("eating_out", 9000)], an.spend_by_sub(an.in_month(spend, first)), TODAY)
    return rules.run_rules(rules.UserCtx(today=TODAY, month=first, txs=spend, goal_statuses=st, income=income))


def test_demo_data_is_valid_deterministic_and_spans_four_months():
    items = demo.flat_items(demo.generate(TODAY))
    assert all(tax.is_valid_sub(i["subcategory"]) and i["amount"] > 0 for i in items)
    assert all(i["date"] <= TODAY for i in items)
    assert {(i["date"].year, i["date"].month) for i in items} == {(2026, 7), (2026, 8), (2026, 9), (2026, 10)}
    assert demo.generate(TODAY) == demo.generate(TODAY)


def test_demo_receipts_sum_to_their_total():
    for g in demo.generate(TODAY):
        if g["receipt"]:
            assert abs(sum(i["amount"] for i in g["items"]) - g["receipt"]["total"]) < 0.01


def test_demo_works_for_any_day_of_month():
    for d in (D(2026, 1, 1), D(2026, 2, 28), D(2026, 3, 31), D(2026, 12, 15)):
        assert all(i["date"] <= d for i in demo.flat_items(demo.generate(d)))


def test_demo_shows_the_scenario_cards():
    cards = demo_cards()
    got = {c["rule_id"] for c in cards}
    assert {"R03_category_growth", "R05_price_increase", "R07_coffee_to_go", "R10_subscriptions_found",
            "R16_reference_shares"} <= got
    delivery = [c for c in cards if c["rule_id"] == "R03_category_growth" and c["category"] == "eating_out"]
    assert delivery and delivery[0]["severity"] == "warning"


def test_demo_subscriptions_are_recognised():
    rec = an.detect_recurring(an.spending(demo_txs()), TODAY)
    assert {r.title for r in rec if r.category == "subscriptions"} == {
        "Кино онлайн подписка", "Облачное хранилище 200 ГБ", "Музыкальный сервис подписка"}
    assert any(r.title == "Оплата аренды квартиры" for r in rec)


def test_demo_history_has_three_complete_months():
    from app.services import history
    rows = [(t.date, t.amount) for t in an.spending(demo_txs())]
    totals = history.monthly_totals(rows, TODAY)
    assert [t.complete for t in totals] == [True, True, True, False]
    assert history.typical_month(totals).n_months == 3
