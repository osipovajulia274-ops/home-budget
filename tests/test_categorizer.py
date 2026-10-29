import numpy as np
import pytest

from app.services import taxonomy as tax
from app.services.categorizer import Categorizer, get_categorizer
from app.services.importing import csv_preview, receipt_preview
from pathlib import Path

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


@pytest.fixture(scope="module")
def cat():
    return get_categorizer()


def test_uses_the_model_chosen_in_stage_2(cat):
    assert cat.kind == "tfidf_logreg_baseline" and cat.threshold == 0.5
    assert cat.labels == list(tax.subcategory_titles())          # same order and set as the taxonomy
    assert cat.labels[-2:] == ["other.unknown", "other.transfers"]


def test_clear_titles(cat):
    got = {t: p.subcategory for t, p in zip(
        ["МОЛОКО ПАСТ. 3,2% 900МЛ", "Яндекс Такси поездка", "Оплата ЖКУ октябрь", "Аптека Ригла №45"],
        cat.categorize(["МОЛОКО ПАСТ. 3,2% 900МЛ", "Яндекс Такси поездка", "Оплата ЖКУ октябрь", "Аптека Ригла №45"]))}
    assert got == {"МОЛОКО ПАСТ. 3,2% 900МЛ": "food.dairy", "Яндекс Такси поездка": "transport.taxi",
                   "Оплата ЖКУ октябрь": "housing.utilities", "Аптека Ригла №45": "health.pharmacy"}


def test_low_confidence_becomes_unknown_and_needs_review(cat):
    p = cat.categorize(["asdf qwer zxcv"])[0]
    assert p.subcategory == tax.UNKNOWN and p.confidence < 0.5 and p.needs_review
    assert p.alternatives                                        # the guess is still offered


def test_user_correction_beats_the_model_and_is_exact_on_normalised_title(cat):
    fb = {"netflix подписка": "subscriptions.streaming"}
    p = cat.categorize(["NETFLIX Подписка"], fb)[0]
    assert (p.subcategory, p.source, p.confidence) == ("subscriptions.streaming", "user", 1.0)
    # digits are normalised away, so a price/article change does not lose the correction
    fb2 = {"молоко 0%": "food.drinks"}
    assert cat.categorize(["Молоко 3%"], fb2)[0].subcategory == "food.drinks"


def test_transfer_rule(cat):
    p = cat.categorize(["Перевод на свою карту"])[0]
    assert (p.subcategory, p.source) == (tax.TRANSFERS, "rule")


def test_empty_and_batch_shapes(cat):
    assert cat.categorize([]) == []
    assert cat.predict_proba(["а", "б", "в"]).shape == (3, 48)
    assert np.allclose(cat.predict_proba(["молоко"]).sum(), 1.0, atol=1e-3)


def test_invalid_feedback_label_is_ignored(cat):
    p = cat.categorize(["Бананы"], {"бананы": "no.such"})[0]
    assert p.source == "model"


def test_csv_preview_uses_mcc_when_model_is_unsure():
    p = csv_preview((SAMPLES / "bank_sample.csv").read_bytes(), Categorizer.load())
    by_title = {i["title"]: i for i in p["items"]}
    taxi = by_title["YANDEX*TAXI MOSCOW RU"]
    assert (taxi["subcategory"], taxi["category_source"], taxi["needs_review"]) == ("transport.taxi", "bank", True)
    assert by_title["Перевод на свою карту"]["subcategory"] == tax.TRANSFERS
    assert p["items_sum"] == round(sum(i["amount"] for i in p["items"] if i["subcategory"] != tax.TRANSFERS), 2)
    assert p["income_total"] == 85000.0


def test_receipt_preview_shape(cat):
    p = receipt_preview((SAMPLES / "receipt_1.txt").read_text(encoding="utf-8"), cat)
    assert p["sum_matches_total"] is True and p["total"] == 1149.59 and len(p["items"]) == 7
    first = p["items"][0]
    assert set(first) >= {"title", "qty", "price", "amount", "subcategory", "category", "confidence",
                          "category_source", "needs_review", "alternatives"}
