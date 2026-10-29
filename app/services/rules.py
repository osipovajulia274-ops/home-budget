"""Recommendation engine (guide section 9, reworked after the 2026-10-28 decision "no forecast as fact").

Each rule is `(UserCtx) -> list[card]`; a card is a dict
    {rule_id, category, severity, title, body, action, payload, effect}
where `effect` (roubles) is only used for ordering and `payload["key"]` identifies the card for "hide".

Differences from the guide (recorded in PROGRESS.md):
  * R01/R02 on the forecast are replaced by the goal cards R01_goal_pace / R02_goal_watch / R02_goal_exceeded
    (see goals.py): they state the facts, not a prediction.
  * R03 compares the month so far with what the user had spent by the same day in earlier months
    (no extrapolation of the current month).
  * R11 cannot see gym visits (the app has no such data), so it only points out the recurring charge.

Tone: calm, no blame. Not financial advice: the cards describe the user's own spending and give everyday hints.
"""
from __future__ import annotations

import calendar
import datetime as dt
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from app.services import analytics as an
from app.services import taxonomy as tax
from app.services.analytics import Tx
from app.services.goals import GoalStatus, goal_recommendations
from app.services.history import fmt_rub, monthly_totals, shift_month, first_of_month

MAX_ACTIVE = 5
SEVERITY_ORDER = {"alert": 0, "warning": 1, "info": 2}
MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
              "ноября", "декабря"]
MONTHS_PREP = ["январе", "феврале", "марте", "апреле", "мае", "июне", "июле", "августе", "сентябре", "октябре",
               "ноябре", "декабре"]
ROSSTAT_NOTE = ("Росстат, выборочное обследование бюджетов домашних хозяйств: на продукты питания в среднем "
                "уходит около трети потребительских расходов (по регионам 31–34%).")


def plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(n)
    if 11 <= n % 100 <= 14:
        return many
    return one if n % 10 == 1 else few if 2 <= n % 10 <= 4 else many


def pct(x: float) -> str:
    return f"{round(x * 100)}%"


@dataclass
class UserCtx:
    today: dt.date                     # the real "today"
    month: dt.date                     # first day of the month being looked at
    txs: Sequence[Tx]                  # ALL spending transactions of the user (no transfers)
    goal_statuses: Sequence[GoalStatus] = ()
    income: Optional[float] = None
    titles: Mapping[str, str] = field(default_factory=tax.all_titles)

    @property
    def as_of(self) -> dt.date:
        return an.as_of_for(self.month, self.today)

    @property
    def cur(self) -> List[Tx]:
        return [t for t in an.in_month(self.txs, self.month) if t.date <= self.as_of]

    def title_of(self, scope: str) -> str:
        return self.titles.get(scope, scope)


def _scope(txs: Sequence[Tx], scope: str) -> List[Tx]:
    return [t for t in txs if an.matches(t.subcategory, scope)]


def _complete_months(ctx: UserCtx, n: int = 3) -> List[dt.date]:
    """First days of the last `n` complete months before the month under review (user-level, like history.py)."""
    rows = [(t.date, t.amount) for t in ctx.txs if t.date <= ctx.as_of]
    totals = monthly_totals(rows, ctx.as_of, n_months=n + 1)
    cur_key = an.month_key(ctx.month)
    return [dt.date(int(m.month[:4]), int(m.month[5:]), 1) for m in totals if m.complete and m.month != cur_key][-n:]


def _by_day(txs: Sequence[Tx], first: dt.date, day: int) -> float:
    cutoff = min(day, calendar.monthrange(first.year, first.month)[1])
    return sum(t.amount for t in txs if t.date.year == first.year and t.date.month == first.month and t.date.day <= cutoff)


def _pace(ctx: UserCtx, scope: str) -> Optional[Tuple[float, float, int]]:
    """(spent so far, median of what was spent by the same day earlier, number of months compared)."""
    months = _complete_months(ctx)
    if not months:
        return None
    rows = _scope(ctx.txs, scope)
    now = sum(t.amount for t in rows if an.in_month([t], ctx.month) and t.date <= ctx.as_of)
    usual = statistics.median(_by_day(rows, m, ctx.as_of.day) for m in months)
    return now, usual, len(months)


def _card(rule_id: str, severity: str, title: str, body: str, action: str, category: Optional[str] = None,
          effect: float = 0.0, key: str = "", **payload) -> dict:
    return {"rule_id": rule_id, "category": category, "severity": severity, "title": title, "body": body,
            "action": action, "effect": float(effect),
            "payload": {"key": f"{rule_id}:{category or ''}:{key}", **payload}}


# ---------------------------------------------------------------------------------- rules ----

def r_goals(ctx: UserCtx) -> List[dict]:
    """R01_goal_pace, R02_goal_watch, R02_goal_exceeded (replace the forecast-based R01/R02)."""
    cards = goal_recommendations(ctx.goal_statuses, ctx.titles)
    for c in cards:
        c["payload"] = {"key": f"{c['rule_id']}:{c['category'] or ''}:{c['payload']['scope']}", **c["payload"]}
    return cards


def r03_category_growth(ctx: UserCtx) -> List[dict]:
    out = []
    for cid in sorted({t.category for t in ctx.cur}):
        p = _pace(ctx, cid)
        if not p:
            continue
        now, usual, n = p
        if usual > 0 and now >= usual * 1.3 and now - usual >= 1000:
            growth = now / usual - 1
            out.append(_card(
                "R03_category_growth", "warning",
                f"«{ctx.title_of(cid)}»: к этому дню на {round(growth * 100)}% больше обычного",
                f"К {ctx.as_of.day}-му числу потрачено {fmt_rub(now)}. В предыдущие {n} "
                f"{plural(n, 'месяц', 'месяца', 'месяцев')} к этому дню в среднем уходило {fmt_rub(usual)} (медиана).",
                f"Поставить лимит на «{ctx.title_of(cid)}»", category=cid, effect=now - usual, key=cid,
                current=round(now, 2), usual=round(usual, 2), growth=round(growth, 3)))
    return out


def r04_sweets_share(ctx: UserCtx) -> List[dict]:
    food = _scope(ctx.cur, "food")
    food_total = an.total_of(food)
    if food_total <= 0:
        return []
    sweets = an.total_of(food, "food.sweets_snacks")
    share = sweets / food_total
    if share <= 0.20:
        return []
    shares = []
    for m in _complete_months(ctx):
        f = an.total_of(_scope(an.in_month(ctx.txs, m), "food"))
        if f > 0:
            shares.append(an.total_of(_scope(an.in_month(ctx.txs, m), "food.sweets_snacks")) / f)
    if shares and share <= statistics.median(shares):
        return []
    return [_card("R04_sweets_share", "info", f"Сладкое и снеки — {pct(share)} продуктовой корзины",
                  f"Из {fmt_rub(food_total)}, потраченных на продукты, на сладкое и снеки ушло {fmt_rub(sweets)}.",
                  "Посмотреть позиции", category="food", effect=sweets, share=round(share, 3))]


def r05_price_increase(ctx: UserCtx) -> List[dict]:
    since = ctx.as_of - dt.timedelta(days=183)
    groups: Dict[str, List[Tx]] = defaultdict(list)
    for t in ctx.txs:
        if since <= t.date <= ctx.as_of and t.category in ("food", "home") and t.title_norm:
            groups[t.title_norm].append(t)
    found = []
    for key, rows in groups.items():
        if len(rows) < 3:
            continue
        rows.sort(key=lambda t: t.date)
        unit = lambda t: t.amount / t.qty if t.qty and t.qty > 0 else t.amount      # noqa: E731
        first, last = unit(rows[0]), unit(rows[-1])
        if first > 0 and last / first - 1 >= 0.15:
            found.append((last / first - 1, rows[0], rows[-1], first, last, key))
    found.sort(key=lambda r: -r[0])
    out = []
    for growth, a, b, first, last, key in found[:2]:
        out.append(_card(
            "R05_price_increase", "info", f"«{b.title}» подорожало на {round(growth * 100)}%",
            f"Цена за единицу: {first:.2f} ₽ ({a.date.strftime('%d.%m.%Y')}) → {last:.2f} ₽ "
            f"({b.date.strftime('%d.%m.%Y')}), покупок за полгода: {len(groups[key])}.".replace(".", ","),
            "Сравнить с аналогами", category=b.category, effect=(last - first) * (b.qty or 1), key=key,
            growth=round(growth, 3), first=round(first, 2), last=round(last, 2)))
    return out


def r06_small_trips(ctx: UserCtx) -> List[dict]:
    trips: Dict[object, float] = defaultdict(float)
    for t in _scope(ctx.cur, "food"):
        trips[("r", t.receipt_id) if t.receipt_id is not None else ("d", t.date)] += t.amount
    n = len(trips)
    if n <= 12:
        return []
    avg = sum(trips.values()) / n
    if avg >= 500:
        return []
    return [_card("R06_small_frequent_trips", "info", f"{n} {plural(n, 'маленький поход', 'маленьких похода', 'маленьких походов')} в магазин",
                  f"Средний чек на продукты — {fmt_rub(avg)}. Крупные закупки раз в неделю обычно выходят дешевле "
                  f"и реже уводят в импульсивные покупки.",
                  "Составить список на неделю", category="food", effect=0.0, trips=n, avg=round(avg, 2))]


def r07_coffee(ctx: UserCtx) -> List[dict]:
    rows = _scope(ctx.cur, "eating_out.coffee")
    if len(rows) < 8:
        return []
    s = an.total_of(rows)
    return [_card("R07_coffee_to_go", "info", f"Кофе с собой: {fmt_rub(s)} за месяц",
                  f"{len(rows)} {plural(len(rows), 'покупка', 'покупки', 'покупок')} за месяц. "
                  f"Если так будет каждый месяц, за год это {fmt_rub(s * 12)} (простое умножение, не прогноз).",
                  "Посмотреть покупки кофе", category="eating_out", effect=s, count=len(rows), total=s)]


def r08_delivery_vs_home(ctx: UserCtx) -> List[dict]:
    food = an.total_of(_scope(ctx.cur, "food"))
    deliv = an.total_of(_scope(ctx.cur, "eating_out.delivery"))
    if food <= 0 or deliv <= 0.15 * food:
        return []
    return [_card("R08_delivery_vs_home", "info", f"Доставка — {pct(deliv / food)} от трат на продукты",
                  f"Доставка еды: {fmt_rub(deliv)}, продукты в магазинах: {fmt_rub(food)}.",
                  "Посмотреть доставку", category="eating_out", effect=deliv, delivery=deliv, food=food)]


def r09_taxi_vs_public(ctx: UserCtx) -> List[dict]:
    taxi = an.total_of(_scope(ctx.cur, "transport.taxi"))
    pub = an.total_of(_scope(ctx.cur, "transport.public"))
    if taxi <= 3000 or taxi <= 3 * pub:
        return []
    return [_card("R09_taxi_vs_public", "info", f"Такси в этом месяце — {fmt_rub(taxi)}",
                  f"На общественный транспорт ушло {fmt_rub(pub)}. Проездной на месяц обычно обходится дешевле "
                  f"ежедневных поездок на такси.",
                  "Сравнить с проездным", category="transport", effect=taxi - pub, taxi=taxi, public=pub)]


def r10_subscriptions(ctx: UserCtx) -> List[dict]:
    subs = [r for r in an.detect_recurring(ctx.txs, ctx.as_of) if r.category == "subscriptions"]
    if len(subs) < 2:
        return []
    month = sum(r.amount for r in subs)
    n = len(subs)
    return [_card("R10_subscriptions_found", "info",
                  f"Найдено {n} {plural(n, 'подписка', 'подписки', 'подписок')}: {fmt_rub(month)} в месяц, "
                  f"{fmt_rub(month * 12)} в год",
                  "Повторяющиеся списания: " + ", ".join(f"{r.title} — {fmt_rub(r.amount)}" for r in subs[:5]) + ".",
                  "Проверить, все ли нужны", category="subscriptions", effect=month * 12, count=n,
                  per_month=round(month, 2))]


def r11_gym_charge(ctx: UserCtx) -> List[dict]:
    gyms = [r for r in an.detect_recurring(_scope(ctx.txs, "health.sport"), ctx.as_of)]
    if not gyms:
        return []
    g = gyms[0]
    return [_card("R11_unused_gym", "info", f"Абонемент: {fmt_rub(g.amount)} в месяц",
                  f"Списание «{g.title}» повторяется каждый месяц. Приложение не видит, ходите ли вы на занятия, — "
                  f"проверьте, пользуетесь ли абонементом.",
                  "Проверить абонемент", category="health", effect=g.amount * 12, key=g.title_norm,
                  per_month=g.amount)]


def r12_pharmacy_growth(ctx: UserCtx) -> List[dict]:
    p = _pace(ctx, "health.pharmacy")
    if not p:
        return []
    now, usual, n = p
    if usual > 0 and now >= usual * 1.5 and now - usual >= 500:
        return [_card("R12_pharmacy_growth", "info", f"Траты в аптеке выросли в {now / usual:.1f} раза".replace(".", ","),
                      f"К {ctx.as_of.day}-му числу потрачено {fmt_rub(now)}, обычно к этому дню — около {fmt_rub(usual)}. "
                      f"Если это лечение — берегите себя.",
                      "Посмотреть покупки", category="health", effect=now - usual, current=now, usual=usual)]
    return []


# (category, month of the peak, what grows)
_SEASONS = [("family_kids", 9, "траты на детей"),
            ("leisure", 12, "траты на развлечения и подарки"),
            ("leisure", 7, "траты на отдых")]


def r13_seasonal(ctx: UserCtx) -> List[dict]:
    out = []
    for cat, peak_month, what in _SEASONS:
        year = ctx.as_of.year if ctx.as_of.month <= peak_month else ctx.as_of.year + 1
        peak_first = dt.date(year, peak_month, 1)
        days_to = (peak_first - ctx.as_of).days
        if not 0 < days_to <= 45:
            continue
        last = dt.date(year - 1, peak_month, 1)
        before = [shift_month(last, -k) for k in (1, 2, 3)]
        have_before = [m for m in before if an.in_month(ctx.txs, m)]
        if not an.in_month(ctx.txs, last) or not have_before:
            continue
        peak = an.total_of(_scope(an.in_month(ctx.txs, last), cat))
        base = statistics.median(an.total_of(_scope(an.in_month(ctx.txs, m), cat)) for m in have_before)
        if peak - base >= 3000:
            said, until = MONTHS_PREP[peak_month - 1], MONTHS_GEN[peak_month - 1]
            out.append(_card("R13_seasonal_prep", "info",
                             f"{what.capitalize()} обычно растут в {said}",
                             f"В прошлом году в {said} на эту категорию ушло на {fmt_rub(peak - base)} больше, чем в "
                             f"предыдущие месяцы. До {until} осталось {days_to} дн. — можно начать откладывать заранее.",
                             "Поставить лимит", category=cat, effect=peak - base, key=str(peak_month),
                             last_year_peak=round(peak, 2), baseline=round(base, 2)))
    return out


_REGULAR_PREFIXES = ("housing.", "subscriptions.")


def r14_anomaly_day(ctx: UserCtx) -> List[dict]:
    since = ctx.as_of - dt.timedelta(days=90)
    daily = an.daily_totals(t for t in ctx.txs if since <= t.date <= ctx.as_of
                            and not t.subcategory.startswith(_REGULAR_PREFIXES))
    if len(daily) < 15:
        return []
    vals = list(daily.values())
    med = statistics.median(vals)
    mad = statistics.median(abs(v - med) for v in vals)
    if mad <= 0:
        return []
    limit = med + 4 * mad
    days = [(v, d) for d, v in daily.items() if v > limit and v >= 3000 and (d.year, d.month) == (ctx.month.year, ctx.month.month)]
    if not days:
        return []
    v, d = max(days)
    return [_card("R14_anomaly_day", "info", f"{d.day} {MONTHS_GEN[d.month - 1]} — необычно крупная трата: {fmt_rub(v)}",
                  f"Обычно за день уходит около {fmt_rub(med)}. Возможно, это плановая покупка — "
                  f"стоит убедиться, что категория определена верно.",
                  "Проверить категорию", effect=v - med, key=d.isoformat(), day=d.isoformat(), amount=round(v, 2))]


def r15_unknown(ctx: UserCtx) -> List[dict]:
    n = len(_scope(ctx.cur, tax.UNKNOWN))
    if n < 5:
        return []
    return [_card("R15_unknown_items", "info", f"{n} {plural(n, 'покупка', 'покупки', 'покупок')} без категории",
                  "Уточните категорию — приложение запомнит ваш выбор для таких же названий.",
                  "Открыть список", category="other", effect=0.0, count=n)]


def r16_reference_shares(ctx: UserCtx) -> List[dict]:
    if not ctx.income or ctx.income <= 0:
        return []
    months = _complete_months(ctx, 1)
    if not months:
        return []
    m = months[-1]
    rows = an.in_month(ctx.txs, m)
    total = an.total_of(rows)
    if total <= 0:
        return []
    food = an.total_of(_scope(rows, "food"))
    return [_card("R16_reference_shares", "info", f"Продукты — {pct(food / total)} ваших трат за {m.strftime('%m.%Y')}",
                  f"Для справки. {ROSSTAT_NOTE} Всего за месяц потрачено {fmt_rub(total)}, это {pct(total / ctx.income)} "
                  f"указанного дохода. Это ориентир, а не оценка.",
                  "Посмотреть категории", category="food", effect=0.0, food_share=round(food / total, 3),
                  month=an.month_key(m))]


RULES: List[Callable[[UserCtx], List[dict]]] = [
    r_goals, r03_category_growth, r04_sweets_share, r05_price_increase, r06_small_trips, r07_coffee,
    r08_delivery_vs_home, r09_taxi_vs_public, r10_subscriptions, r11_gym_charge, r12_pharmacy_growth,
    r13_seasonal, r14_anomaly_day, r15_unknown, r16_reference_shares,
]


def run_rules(ctx: UserCtx) -> List[dict]:
    """All cards that fire, most important first: alert > warning > info, then by the rouble effect."""
    cards: List[dict] = []
    for rule in RULES:
        cards.extend(rule(ctx))
    cards.sort(key=lambda c: (SEVERITY_ORDER[c["severity"]], -c["effect"]))
    return cards


def pick_active(cards: Sequence[dict], hidden_keys: Sequence[str] = (), limit: int = MAX_ACTIVE) -> List[dict]:
    """Drop the cards the user has hidden this month, keep at most `limit`."""
    hidden = set(hidden_keys)
    return [c for c in cards if c["payload"]["key"] not in hidden][:limit]
