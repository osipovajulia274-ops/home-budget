"""API smoke tests (guide 8.6): import -> confirm -> correction -> summary -> history -> goals -> recommendations.

They need fastapi, sqlalchemy and httpx (`pip install -r requirements-dev.txt`) and are skipped without them.
"""
import datetime as dt
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("httpx")

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
TODAY = dt.date(2026, 10, 29)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite://")             # one in-memory database per test
    monkeypatch.setenv("APP_ENV", "test")
    from app import config, db
    from app.services import clock

    config.get_settings.cache_clear()
    db.reset_engine()
    monkeypatch.setattr(clock, "today", lambda tz="Europe/Moscow": TODAY)
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        yield c
    db.reset_engine()
    config.get_settings.cache_clear()


def receipt(name="receipt_1.txt"):
    return (SAMPLES / name).read_text(encoding="utf-8")


def confirm_body(preview, overrides=None):
    items = [{"title": i["title"], "qty": i["qty"], "amount": i["amount"], "subcategory": i["subcategory"],
              "predicted_subcategory": i["subcategory"], "confidence": i["confidence"],
              "category_source": i["category_source"]} for i in preview["items"]]
    for idx, sub in (overrides or {}).items():
        items[idx]["subcategory"] = sub
    return {"source": "text", "store": preview["store"], "purchased_at": preview["purchased_at"],
            "total": preview["total"], "sum_matches_total": preview["sum_matches_total"], "items": items}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["models"]["classifier"] is True


def test_taxonomy(client):
    t = client.get("/api/taxonomy").json()
    assert len(t) == 13 and sum(len(c["subcategories"]) for c in t) == 48


def test_receipt_flow_preview_does_not_save_then_confirm_saves(client):
    p = client.post("/api/import/receipt-text", json={"text": receipt()}).json()
    assert p["sum_matches_total"] is True and len(p["items"]) == 7
    assert client.get("/api/transactions", params={"month": "2026-10"}).json() == []      # nothing saved yet
    saved = client.post("/api/import/confirm", json=confirm_body(p)).json()
    assert saved["saved"] == 7 and saved["receipt_id"]
    txs = client.get("/api/transactions", params={"month": "2026-10"}).json()
    assert len(txs) == 7 and all(t["date"] == "2026-10-27" for t in txs)
    s = client.get("/api/summary", params={"month": "2026-10"}).json()
    assert s["total"] == 1149.59 and abs(sum(c["share"] for c in s["by_category"]) - 1) < 1e-3


def test_correction_in_preview_is_remembered_on_reimport(client):
    p = client.post("/api/import/receipt-text", json={"text": receipt()}).json()
    milk = next(i for i, it in enumerate(p["items"]) if it["title"].startswith("МОЛОКО"))
    client.post("/api/import/confirm", json=confirm_body(p, {milk: "food.drinks"}))
    again = client.post("/api/import/receipt-text", json={"text": receipt()}).json()
    assert again["items"][milk]["subcategory"] == "food.drinks"
    assert again["items"][milk]["category_source"] == "user" and again["items"][milk]["confidence"] == 1.0


def test_patch_transaction_records_feedback_and_applies_to_next_import(client):
    p = client.post("/api/import/receipt-text", json={"text": receipt()}).json()
    client.post("/api/import/confirm", json=confirm_body(p))
    tx = next(t for t in client.get("/api/transactions", params={"month": "2026-10"}).json() if t["title"].startswith("СЫР"))
    r = client.patch(f"/api/transactions/{tx['id']}", json={"subcategory": "food.sweets_snacks"})
    assert r.status_code == 200 and r.json()["subcategory"] == "food.sweets_snacks" and r.json()["category"] == "food"
    assert r.json()["category_source"] == "user"
    again = client.post("/api/import/receipt-text", json={"text": receipt()}).json()
    assert next(i for i in again["items"] if i["title"].startswith("СЫР"))["subcategory"] == "food.sweets_snacks"
    assert client.patch(f"/api/transactions/{tx['id']}", json={"subcategory": "nope"}).status_code == 422
    assert client.patch("/api/transactions/999999", json={"subcategory": "food.dairy"}).status_code == 404


def test_manual_transaction_delete_and_validation(client):
    r = client.post("/api/transactions", json={"date": "2026-10-05", "title": "Яндекс Такси поездка", "amount": 350})
    assert r.status_code == 201 and r.json()["subcategory"] == "transport.taxi"
    assert client.post("/api/transactions", json={"date": "2026-10-05", "title": "x", "amount": -5}).status_code == 422
    assert client.post("/api/transactions", json={"date": "2026-10-05", "title": "x", "amount": 5,
                                                  "subcategory": "bad"}).status_code == 422
    assert client.delete(f"/api/transactions/{r.json()['id']}").status_code == 204
    assert client.get("/api/transactions", params={"month": "2026-10"}).json() == []


def test_csv_preview_and_confirm_excludes_transfers_from_spending(client):
    files = {"file": ("bank.csv", (SAMPLES / "bank_sample.csv").read_bytes(), "text/csv")}
    p = client.post("/api/import/csv", files=files).json()
    assert p["encoding"] == "cp1251" and p["income_total"] == 85000.0 and len(p["items"]) == 19
    body = {"source": "csv", "items": [{"title": i["title"], "amount": i["amount"], "date": i["date"],
                                        "subcategory": i["subcategory"], "predicted_subcategory": i["subcategory"],
                                        "confidence": i["confidence"], "category_source": i["category_source"]}
                                       for i in p["items"]]}
    assert client.post("/api/import/confirm", json=body).json() == {"receipt_id": None, "saved": 19}
    s = client.get("/api/summary", params={"month": "2026-10"}).json()
    assert s["total"] == pytest.approx(p["items_sum"]) and "other" not in {c["category"] for c in s["by_category"]}


def test_csv_errors_are_422(client):
    r = client.post("/api/import/csv", files={"file": ("x.csv", b"", "text/csv")})
    assert r.status_code == 422 and "пустой" in r.json()["detail"]
    r = client.post("/api/import/csv", files={"file": ("x.csv", "a;b\n1;2\n".encode(), "text/csv")},
                    data={"mapping": "not json"})
    assert r.status_code == 422


def test_confirm_rejects_unknown_subcategory(client):
    body = {"source": "text", "items": [{"title": "x", "amount": 1, "subcategory": "no.such"}]}
    assert client.post("/api/import/confirm", json=body).status_code == 422


def test_demo_scenario_history_goals_recommendations(client):
    assert client.post("/api/demo/load").status_code == 204
    s = client.get("/api/summary").json()
    assert s["month"] == "2026-10" and s["total"] > 0 and s["prev_month_total"] > 0

    h = client.get("/api/history").json()
    assert [m["complete"] for m in h["months"]] == [True, True, True, False]
    assert h["typical"]["n_months"] == 3 and h["pace"]["usual_by_now"] > 0 and h["note"] is None

    g = client.post("/api/goals", json={"scope": "eating_out", "limit_amount": 9000})
    assert g.status_code == 201
    assert client.post("/api/goals", json={"scope": "eating_out", "limit_amount": 1}).status_code == 409
    assert client.post("/api/goals", json={"scope": "nope", "limit_amount": 1}).status_code == 422
    goals = client.get("/api/goals").json()
    assert goals[0]["level"] == "exceeded" and goals[0]["over_by"] > 0
    assert client.put(f"/api/goals/{g.json()['id']}", json={"scope": "eating_out", "limit_amount": 50000}).status_code == 200
    assert client.get("/api/goals").json()[0]["level"] in ("ok", "ahead_of_pace")

    recs = client.get("/api/recommendations").json()
    assert 1 <= len(recs) <= 5
    assert any(r["rule_id"] == "R03_category_growth" for r in recs + client.get("/api/recommendations").json())
    assert all("прогноз" not in (r["title"] + r["body"]).lower() for r in recs)

    subs = client.get("/api/subscriptions").json()
    assert subs["subscriptions_per_month"] == 847.0 and subs["subscriptions_per_year"] == 10164.0


def test_dismissed_card_does_not_return_this_month(client):
    client.post("/api/demo/load")
    recs = client.get("/api/recommendations").json()
    victim = recs[0]
    assert client.post(f"/api/recommendations/{victim['id']}/dismiss").status_code == 204
    after = client.get("/api/recommendations").json()
    assert victim["id"] not in [r["id"] for r in after]
    assert victim["title"] not in [r["title"] for r in after]
    assert client.post("/api/recommendations/999999/dismiss").status_code == 404


def test_history_before_first_full_month_gives_no_comparison(client):
    client.post("/api/transactions", json={"date": "2026-10-20", "title": "Бананы", "amount": 100})
    h = client.get("/api/history").json()
    assert h["typical"] is None and h["pace"] is None and "первого полного месяца" in h["note"]


def test_users_are_isolated_by_cookie(client):
    from fastapi.testclient import TestClient
    from app.main import app

    client.post("/api/transactions", json={"date": "2026-10-05", "title": "Чай", "amount": 100})
    with TestClient(app) as other:
        assert other.get("/api/transactions", params={"month": "2026-10"}).json() == []
    assert len(client.get("/api/transactions", params={"month": "2026-10"}).json()) == 1


def test_delete_my_data_clears_everything(client):
    client.post("/api/demo/load")
    client.post("/api/goals", json={"scope": "total", "limit_amount": 1000})
    client.get("/api/recommendations")
    assert client.post("/api/me/delete").status_code == 204
    from app import models
    from app.db import get_engine
    from sqlalchemy import func, select
    from sqlalchemy.orm import Session

    with Session(get_engine()) as s:
        for model in (models.Transaction, models.Receipt, models.Goal, models.Recommendation,
                      models.CategoryFeedback, models.User):
            assert s.scalar(select(func.count()).select_from(model)) == 0, model.__tablename__


def test_bad_month_param(client):
    assert client.get("/api/summary", params={"month": "2026-13"}).status_code == 422
