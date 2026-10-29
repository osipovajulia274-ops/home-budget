from pathlib import Path

import pytest

from app.services.csv_import import CsvImportError, parse_csv, parse_amount, parse_date

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


def test_cp1251_semicolon_bank_file():
    r = parse_csv((SAMPLES / "bank_sample.csv").read_bytes())
    assert (r.encoding, r.delimiter) == ("cp1251", ";")
    assert r.mapping["date"] == "Дата операции" and r.mapping["amount"] == "Сумма операции"
    assert r.mapping["category"] == "Категория" and r.mapping["mcc"] == "MCC"
    assert len(r.rows) == 19 and r.income_count == 1 and r.income_total == 85000.0
    assert r.rows[0].amount == 1249.5 and r.rows[0].mcc == "5411" and r.rows[0].bank_category == "Супермаркеты"
    assert [x.title for x in r.rows if x.is_transfer] == ["Перевод на свою карту", "Перевод СБП себе"]


def test_utf8_comma_file_merges_merchant_and_description():
    r = parse_csv((SAMPLES / "bank_sample_2.csv").read_bytes())
    assert (r.encoding, r.delimiter) == ("utf-8-sig", ",")
    assert r.mapping["description"] == ["merchant", "description"]
    assert r.rows[1].title == "Яндекс Такси Поездка" and len(r.rows) == 9


def test_user_mapping_overrides_guess():
    data = "когда;сколько;что\n01.10.2026;-100,00;Чай\n".encode("utf-8")
    with pytest.raises(CsvImportError):
        parse_csv(data)                                    # nothing is recognised
    r = parse_csv(data, {"date": "когда", "amount": "сколько", "description": ["что"]})
    assert r.rows[0].title == "Чай" and r.rows[0].amount == 100.0


def test_errors_are_readable():
    with pytest.raises(CsvImportError, match="пустой"):
        parse_csv(b"")
    with pytest.raises(CsvImportError, match="нет строк"):
        parse_csv("date;amount;description\n".encode())


def test_bad_rows_are_skipped_and_counted():
    data = "date;amount;description\n01.10.2026;-10,00;Чай\nвчера;-5;Кофе\n02.10.2026;abc;Хлеб\n".encode()
    r = parse_csv(data)
    assert len(r.rows) == 1 and r.skipped == 2 and any("2" in w for w in r.warnings)


def test_phone_topup_is_spending_not_a_transfer():
    r = parse_csv("date;amount;description\n01.10.2026;-300;Пополнение мобильного счёта\n".encode())
    assert r.rows[0].is_transfer is False


@pytest.mark.parametrize("s,v", [("-1 249,50", -1249.5), ("1,234.56", 1234.56), ("1.234,56", 1234.56),
                                 ("−100", -100.0), ("85000,00 ₽", 85000.0), ("", None), ("abc", None)])
def test_parse_amount(s, v):
    assert parse_amount(s) == v


def test_parse_date_formats():
    assert str(parse_date("01.10.2026 09:12:44")) == "2026-10-01"
    assert str(parse_date("2026-10-01")) == "2026-10-01"
    assert parse_date("31.02.2026") is None
