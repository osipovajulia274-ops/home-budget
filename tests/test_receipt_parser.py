import datetime as dt
import json
from pathlib import Path

import pytest

from app.services.receipt_parser import parse_receipt

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
EXPECTED = json.loads((SAMPLES / "expected.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_samples_match_expected(name):
    r = parse_receipt((SAMPLES / name).read_text(encoding="utf-8"))
    e = EXPECTED[name]
    assert r.store == e["store"]
    assert r.purchased_at.isoformat() == e["purchased_at"]
    assert r.total == e["total"]
    assert [(i.title, i.qty, i.amount) for i in r.items] == [(i["title"], i["qty"], i["amount"]) for i in e["items"]]
    assert r.sum_matches_total is True and r.warnings == []


def test_discount_is_subtracted_from_previous_item():
    r = parse_receipt("ООО Тест\n-----\nЧипсы 129.00\nСКИДКА -30.00\n-----\nИТОГ 99.00")
    assert [(i.title, i.amount) for i in r.items] == [("Чипсы", 99.0)]
    assert r.sum_matches_total is True


def test_name_on_one_line_weight_on_next():
    r = parse_receipt("ООО Тест\n-----\nБананы\n0,762 кг x 119,90 = 91,36\n-----\nИТОГ 91,36")
    assert r.items[0].qty == 0.762 and r.items[0].amount == 91.36


def test_dotted_leader_before_price():
    r = parse_receipt("ООО Тест\n-----\nКефир 1л.........89.90\n-----\nИТОГ 89.90")
    assert [(i.title, i.amount) for i in r.items] == [("Кефир 1л", 89.9)]


def test_mismatch_gives_warning_and_flag():
    r = parse_receipt("ООО Тест\n-----\nМолоко 2 x 89.90 =179.80\n-----\nИТОГ 200.00")
    assert r.sum_matches_total is False
    assert any("не совпала" in w and "20,20" in w for w in r.warnings)


def test_missing_total_warns():
    r = parse_receipt("Хлеб 54.00")
    assert r.total is None and r.sum_matches_total is None
    assert any("ИТОГ" in w for w in r.warnings)


def test_sole_proprietor_and_inn_are_not_kept():
    r = parse_receipt('ИП Сидоров Алексей Петрович ИНН 123456789012\n-----\nЧай 100.00\n-----\nИТОГ 100.00')
    assert r.store == "ИП С"
    r2 = parse_receipt('ООО "Магазин" ИНН 7700000000\n-----\nЧай 100.00\n-----\nИТОГ 100.00')
    assert "7700000000" not in r2.store and "ИНН" not in r2.store


def test_service_lines_are_not_items():
    r = parse_receipt("ООО Тест\n-----\nЧай 100.00\nНДС 20% 16.67\nСДАЧА 50.00\nКАССИР Иванова 1.00\n-----\nИТОГ 100.00")
    assert [i.title for i in r.items] == ["Чай"]


def test_date_variants_and_default():
    assert parse_receipt("Дата: 12.10.26 13:25\n-----\nЧай 10.00").purchased_at == dt.datetime(2026, 10, 12, 13, 25)
    assert parse_receipt("05.11.2026\n-----\nЧай 10.00").purchased_at == dt.datetime(2026, 11, 5)
    assert parse_receipt("Чай 10.00", dt.date(2026, 10, 1)).purchased_at == dt.datetime(2026, 10, 1)


def test_garbage_gives_no_items_and_a_warning():
    r = parse_receipt("просто какой-то текст")
    assert r.items == [] and any("позиции" in w for w in r.warnings)
