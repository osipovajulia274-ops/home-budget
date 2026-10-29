"""Fallback mapping of a bank's own category / MCC code to our taxonomy (guide 8.4).

Used only when the model is unsure (confidence < 0.5). These are coarse hand-written mappings: a bank category
like "Супермаркеты" cannot say whether the purchase was dairy or sweets, so it lands in the broadest class.
"""
from __future__ import annotations

from typing import Optional

from app.services.normalize import normalize

# normalised bank category name (substring) -> our subcategory; first match wins, order matters
_BANK_CATEGORIES = [
    ("перевод", "other.transfers"),
    ("супермаркет", "food.grocery"),
    ("продукт", "food.grocery"),
    ("фастфуд", "eating_out.fastfood"),
    ("рестора", "eating_out.cafe"),
    ("кафе", "eating_out.cafe"),
    ("доставка еды", "eating_out.delivery"),
    ("такси", "transport.taxi"),
    ("азс", "transport.fuel"),
    ("топлив", "transport.fuel"),
    ("авто", "transport.car_service"),
    ("парковк", "transport.car_service"),
    ("транспорт", "transport.public"),
    ("аптек", "health.pharmacy"),
    ("медицин", "health.doctors"),
    ("здоровь", "health.pharmacy"),
    ("спорт", "health.sport"),
    ("жкх", "housing.utilities"),
    ("коммунал", "housing.utilities"),
    ("связь", "housing.internet_mobile"),
    ("интернет", "housing.internet_mobile"),
    ("аренд", "housing.rent"),
    ("ипотек", "housing.rent"),
    ("подписк", "subscriptions.other"),
    ("кино", "leisure.events"),
    ("развлечен", "leisure.events"),
    ("путешеств", "leisure.travel"),
    ("отел", "leisure.travel"),
    ("игр", "leisure.games"),
    ("одежд", "clothes_beauty.clothes"),
    ("обув", "clothes_beauty.clothes"),
    ("красот", "clothes_beauty.salons"),
    ("косметик", "clothes_beauty.cosmetics"),
    ("дом", "home.household"),
    ("ремонт", "home.repair"),
    ("электроник", "home.appliances"),
    ("техник", "home.appliances"),
    ("дет", "family_kids.goods"),
    ("образован", "education.courses"),
    ("книг", "education.books"),
    ("животн", "pets.goods"),
    ("питомц", "pets.goods"),
    ("ветеринар", "pets.vet"),
]

_MCC = {
    5411: "food.grocery", 5499: "food.grocery", 5422: "food.meat_fish", 5441: "food.sweets_snacks",
    5451: "food.dairy", 5462: "food.bakery", 5921: "food.alcohol",
    5812: "eating_out.cafe", 5813: "eating_out.cafe", 5814: "eating_out.fastfood",
    4121: "transport.taxi", 4111: "transport.public", 4112: "transport.public", 4131: "transport.public",
    5541: "transport.fuel", 5542: "transport.fuel", 7523: "transport.car_service", 7538: "transport.car_service",
    5912: "health.pharmacy", 8011: "health.doctors", 8021: "health.doctors", 8062: "health.doctors",
    7997: "health.sport", 7941: "health.sport",
    4900: "housing.utilities", 4814: "housing.internet_mobile", 4899: "housing.internet_mobile",
    7832: "leisure.events", 7922: "leisure.events", 7993: "leisure.games", 7011: "leisure.travel",
    4722: "leisure.travel", 3000: "leisure.travel",
    5651: "clothes_beauty.clothes", 5661: "clothes_beauty.clothes", 5977: "clothes_beauty.cosmetics",
    7230: "clothes_beauty.salons", 7298: "clothes_beauty.salons",
    5200: "home.repair", 5211: "home.repair", 5712: "home.household", 5732: "home.appliances",
    5734: "home.appliances", 5311: "home.household", 5331: "home.household",
    5641: "family_kids.goods", 5945: "leisure.hobby", 5942: "education.books", 8299: "education.courses",
    8211: "family_kids.school", 8220: "education.courses", 5995: "pets.goods", 742: "pets.vet",
    5968: "subscriptions.other", 4816: "subscriptions.cloud_apps", 6012: "other.transfers", 6538: "other.transfers",
}


def from_bank_category(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = normalize(name)
    for key, sub in _BANK_CATEGORIES:
        if key in n:
            return sub
    return None


def from_mcc(mcc) -> Optional[str]:
    try:
        return _MCC.get(int(str(mcc).strip()))
    except (TypeError, ValueError):
        return None
