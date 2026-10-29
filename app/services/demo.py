"""Demo data for the "Загрузить демо-данные" button: three full months plus the current month of a fictional
couple. SYNTHETIC: titles and amounts are invented, nothing here is a measurement of real households.

Pure Python (no pandas/numpy) so that it runs on Render. The same user always gets the same data for the same
`today` (fixed seed), which keeps screenshots and tests reproducible.
"""
from __future__ import annotations

import calendar
import datetime as dt
import random
from typing import Dict, List

from app.services.history import shift_month

DEMO_STORE = "Демо-магазин (синтетические данные)"

# (title, subcategory, low price, high price, weighed?)
GROCERIES = [
    ("Сметана 20% 300г", "food.dairy", 78, 96, False),
    ("Творог 5% 200г", "food.dairy", 69, 88, False),
    ("Яйцо куриное С1 10шт", "food.dairy", 92, 118, False),
    ("Куриное филе охлажденное", "food.meat_fish", 280, 420, True),
    ("Фарш свино-говяжий", "food.meat_fish", 300, 450, True),
    ("Филе горбуши", "food.meat_fish", 380, 560, True),
    ("Бананы", "food.produce", 90, 130, True),
    ("Яблоки Гала", "food.produce", 95, 150, True),
    ("Огурцы короткоплодные", "food.produce", 140, 260, True),
    ("Хлеб бородинский нарезка", "food.bakery", 48, 72, False),
    ("Батон нарезной", "food.bakery", 38, 58, False),
    ("Макароны спагетти 450г", "food.grocery", 55, 90, False),
    ("Гречка ядрица 800г", "food.grocery", 80, 130, False),
    ("Масло подсолнечное 1л", "food.grocery", 120, 170, False),
    ("Шоколад молочный 90г", "food.sweets_snacks", 70, 110, False),
    ("Чипсы картофельные 140г", "food.sweets_snacks", 95, 140, False),
    ("Печенье овсяное 300г", "food.sweets_snacks", 60, 100, False),
    ("Вода питьевая негазированная 1,5л", "food.drinks", 32, 48, False),
    ("Чай черный 100 пакетиков", "food.drinks", 140, 230, False),
    ("Средство для мытья посуды 450мл", "home.chemicals", 85, 140, False),
    ("Пакеты мусорные 30л", "home.household", 55, 95, False),
]
MILK = "Молоко пастеризованное 3,2% 900мл"       # added separately: its price grows month by month


def _plan(today: dt.date) -> List[dt.date]:
    first = today.replace(day=1)
    return [shift_month(first, -3), shift_month(first, -2), shift_month(first, -1), first]


def generate(today: dt.date, seed: int = 42) -> List[dict]:
    """A list of `{"receipt": {...} | None, "items": [{date, title, amount, qty, subcategory}]}`."""
    rng = random.Random(seed)
    out: List[dict] = []

    def single(date: dt.date, title: str, sub: str, amount: float, qty: float = 1.0):
        if date <= today:
            out.append({"receipt": None, "items": [{"date": date, "title": title, "amount": round(amount, 2),
                                                    "qty": qty, "subcategory": sub}]})

    for mi, first in enumerate(_plan(today)):
        dim = calendar.monthrange(first.year, first.month)[1]
        current = first.month == today.month and first.year == today.year
        d = lambda day: first.replace(day=min(day, dim))      # noqa: E731
        k = 1 + 0.004 * mi                                     # a little inflation month to month

        # regular payments (same description and amount, so they are recognised as subscriptions)
        single(d(3), "Оплата аренды квартиры", "housing.rent", 38000)
        single(d(8), "Оплата ЖКУ", "housing.utilities", 5300 * rng.uniform(0.97, 1.03))
        single(d(10), "Мобильная связь тариф", "housing.internet_mobile", 650)
        single(d(12), "Домашний интернет", "housing.internet_mobile", 750)
        single(d(5), "Кино онлайн подписка", "subscriptions.streaming", 399)
        single(d(14), "Облачное хранилище 200 ГБ", "subscriptions.cloud_apps", 149)
        single(d(21), "Музыкальный сервис подписка", "subscriptions.streaming", 299)
        single(d(1), "Абонемент фитнес", "health.sport", 3200)

        # grocery trips: one receipt each, 3-7 items
        for day in sorted(rng.sample(range(1, dim + 1), 13)):
            date = d(day)
            if date > today:
                continue
            items = []
            for title, sub, lo, hi, weighed in rng.sample(GROCERIES, rng.randint(3, 7)):
                price = round(rng.uniform(lo, hi) * k, 2)
                qty = round(rng.uniform(0.3, 1.5), 3) if weighed else float(rng.choice([1, 1, 1, 2]))
                items.append({"date": date, "title": title, "qty": qty, "subcategory": sub,
                              "amount": round(price * (qty if weighed else qty), 2)})
            if rng.random() < 0.7:                                     # the milk gets more expensive
                qty = float(rng.choice([1, 2]))
                items.append({"date": date, "title": MILK, "qty": qty, "subcategory": "food.dairy",
                              "amount": round((82 + 6 * mi) * qty, 2)})
            total = round(sum(i["amount"] for i in items), 2)
            out.append({"receipt": {"store": DEMO_STORE, "purchased_at": dt.datetime(date.year, date.month, date.day, 18, 30),
                                    "total": total}, "items": items})

        for day in sorted(rng.sample(range(1, dim + 1), 9)):
            single(d(day), "Капучино 300мл", "eating_out.coffee", rng.uniform(220, 260) * k)
        for day in sorted(rng.sample(range(1, dim + 1), 8 if current else 3)):
            single(d(day), "Доставка еды заказ", "eating_out.delivery", rng.uniform(900, 1500) * k)
        for day in sorted(rng.sample(range(1, dim + 1), 2)):
            single(d(day), "Бизнес-ланч кафе", "eating_out.cafe", rng.uniform(450, 700) * k)
        for day in sorted(rng.sample(range(1, dim + 1), 4)):
            single(d(day), "Яндекс Такси поездка", "transport.taxi", rng.uniform(250, 600) * k)
        for day in sorted(rng.sample(range(1, dim + 1), 8)):
            single(d(day), "Метро проезд", "transport.public", 57.0)
        for day in sorted(rng.sample(range(1, dim + 1), 2)):
            single(d(day), "Аптека лекарства", "health.pharmacy", rng.uniform(400, 1100) * k)
        single(d(rng.randint(10, 25)), "Билет в кино", "leisure.events", rng.uniform(600, 1100))
        single(d(rng.randint(10, 25)), "Стрижка женская", "clothes_beauty.salons", rng.uniform(1800, 2600))
    return out


def flat_items(plan: List[dict]) -> List[Dict]:
    return [i for p in plan for i in p["items"]]
