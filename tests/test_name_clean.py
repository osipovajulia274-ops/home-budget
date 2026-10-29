"""Name cleaning, language filter, guards, map order and receipt-noise sanity (found in the first Colab run)."""
import random
import re

import pytest

from ml.name_clean import GuardReport, apply_guards, clean_name, process_rows, reject_reason
from ml.off_extract import load_map, map_tags
from ml.receipt_noise import drop_word, noisy_variants, truncate, load_rules, unit_kind_for


# ------------------------------------------------------------------ html / whitespace
@pytest.mark.parametrize("raw,expected", [
    ("Влажные салфетки &quot;Бубочка&quot;", 'Влажные салфетки "Бубочка"'),
    ("БИОЙОГУРТ &QUOT;БИФИЛАЙФ&QUOT; ФРУКТОВЫЙ", 'БИОЙОГУРТ "БИФИЛАЙФ" ФРУКТОВЫЙ'),
    ("Сок &amp;quot;Добрый&amp;quot;", 'Сок "Добрый"'),
    ("  Чай черный \t 100г ", "Чай черный 100г"),
    (None, ""),
])
def test_clean_name(raw, expected):
    assert clean_name(raw) == expected


# ------------------------------------------------------------------ language filter
@pytest.mark.parametrize("name", [
    "Молоко питьевое пастеризованное", "Съёмный чехол для телефона", "Объём 5л вода", "Яйцо белое С1",
    "Felix Аппетитные кусочки с индейкой в желе",
])
def test_russian_names_pass(name):
    assert reject_reason(name) is None


@pytest.mark.parametrize("name,why", [
    ("Хліб вівсяний з льоном", "ukrainian_or_other"),
    ("Пиво світле пастеризоване", "ukrainian_or_other"),
    ("Хляб Пълнозърнест със 100% Лимец", "bulgarian"),
    ("Прясно краве мляко 2%", "bulgarian"),
    ("Бира Туборг", "bulgarian"),
    ("Ръжено-пшеничен хлѣб", "ukrainian_or_other"),
    ("Ab", "short"),
    ("1234 x", "few_cyrillic"),
])
def test_foreign_names_rejected(name, why):
    assert reject_reason(name) == why


def test_country_filter():
    assert reject_reason("Молоко питьевое", "en:bulgaria") == "country"
    assert reject_reason("Молоко питьевое", "en:russia,en:bulgaria") is None
    assert reject_reason("Молоко питьевое", "") is None


# ------------------------------------------------------------------ guards (examples from the real run)
@pytest.mark.parametrize("name,sub,expected", [
    ("Кефир 3,2%", "food.drinks", "food.dairy"),
    ("яйцо белое", "food.alcohol", "food.dairy"),
    ("Окское яйцо мытое", "food.dairy", "food.dairy"),
    ("Протеиновый Йогурт", "food.sweets_snacks", "food.dairy"),
    ("Молоко питьевое пастеризованное", "food.drinks", "food.dairy"),
    ("Масло сливочное 82%", "food.grocery", "food.dairy"),
    ("Комплексная зубная щётка", "clothes_beauty.cosmetics", "health.hygiene"),
    ("Подгузники Pampers 4", "home.household", "family_kids.goods"),
    # must NOT move
    ("Чипсы со вкусом сыра", "food.sweets_snacks", "food.sweets_snacks"),
    ("Мороженое пломбир в вафельном стаканчике", "food.sweets_snacks", "food.sweets_snacks"),
    ("Овсяное молоко", "food.drinks", "food.drinks"),
    ("Йогурт в шоколадной глазури", "food.sweets_snacks", "food.sweets_snacks"),
    ("Пиво Грин бит манго", "food.alcohol", "food.alcohol"),
    ("Солнцезащитные сливки SPF30", "clothes_beauty.cosmetics", "clothes_beauty.cosmetics"),
])
def test_guard_reassign(name, sub, expected):
    assert apply_guards(name, sub)[0] == expected


@pytest.mark.parametrize("name,sub,dropped", [
    ("Гречка ядрица", "pets.food", True),
    ("Felix Аппетитные кусочки с индейкой", "pets.food", False),
    ("Сухой корм для кошек с рисом", "pets.food", False),
    ("Творог БМ 5%", "home.household", True),
    ("Термопакет изотермический 320x350 мм", "home.household", False),
])
def test_guard_drop(name, sub, dropped):
    assert (apply_guards(name, sub)[0] is None) is dropped


def test_process_rows_counts_everything():
    rep = GuardReport()
    rows = [
        ("Кефир &quot;Простоквашино&quot; 1%", "food.drinks", "", "off"),   # cleaned + reassigned
        ("Хляб Пълнозърнест", "food.bakery", "", "off"),                    # bulgarian
        ("Гречка ядрица", "pets.food", "", "opff"),                         # dropped
        ("Батон нарезной", "food.bakery", "", "off"),                       # untouched
    ]
    kept = process_rows(rows, rep)
    assert kept == [('Кефир "Простоквашино" 1%', "food.dairy"), ("Батон нарезной", "food.bakery")]
    assert rep.counts["html_or_whitespace_cleaned"] == 1
    assert rep.counts["off:reject:bulgarian"] == 1
    assert rep.counts["opff:drop:petfood_is_human_food"] == 1
    assert rep.counts["off:reassign:dairy_stems"] == 1


# ------------------------------------------------------------------ tag map order
def test_tag_map_order_fixes():
    food = load_map()["food"]
    assert map_tags(["en:beverages", "en:dairies", "en:dairy-drinks", "en:fermented-milk-products"], food) == "food.dairy"
    assert map_tags(["en:desserts", "en:dairies", "en:yogurts"], food) == "food.dairy"
    assert map_tags(["en:frozen-foods", "en:desserts", "en:dairies", "en:ice-creams"], food) == "food.sweets_snacks"
    assert map_tags(["en:beverages", "en:alcoholic-beverages", "en:beers"], food) == "food.alcohol"
    assert map_tags(["en:beverages", "en:sodas"], food) == "food.drinks"
    assert map_tags(["en:fermented-foods"], food) is None            # sauerkraut / kimchi / kombucha: no longer "dairy"
    assert map_tags(["en:plant-based-foods", "en:cereals-and-potatoes"], food) == "food.grocery"


def test_products_map_has_no_junk_bin():
    products = load_map()["products"]
    assert map_tags(["en:something-unknown"], products) is None
    assert map_tags(["en:batteries"], products) == "home.appliances"


# ------------------------------------------------------------------ receipt noise sanity
# a unit SUFFIX at the end of the line (prefix tokens such as "881785ШТ." are a different thing)
UNIT_RE = re.compile(r"(?:\d\s?(?:г|кг|мл|л|шт)\.?|вес|№\s?\d+|1/\d+)\s*$", re.IGNORECASE)


@pytest.mark.parametrize("sub,clean", [
    ("clothes_beauty.clothes", "Рубашка мужская"),
    ("home.appliances", "Экшн-камера"),
    ("leisure.hobby", "Гитара акустическая"),
    ("home.repair", "Унитаз"),
    ("transport.fuel", "АИ-95 бензин"),
])
def test_no_units_for_non_measured_goods(sub, clean):
    kind = unit_kind_for(sub)
    assert kind == "none"
    for v in noisy_variants(clean, 40, random.Random(3), "goods", unit_kind=kind):
        assert not UNIT_RE.search(v), v


def test_units_match_the_product_kind():
    rules = load_rules()
    assert unit_kind_for("food.drinks", rules) == "volume"
    assert unit_kind_for("food.meat_fish", rules) == "mass"
    for v in noisy_variants("Сок яблочный", 60, random.Random(4), "goods", unit_kind="volume"):
        assert not re.search(r"\d\s?(?:г|кг)\b", v, re.IGNORECASE), v


def test_drop_word_keeps_the_head_noun():
    rng = random.Random(0)
    assert drop_word("Батон на молоке", rng) == "Батон на молоке"            # < 4 words: untouched
    assert drop_word("Трансфер до отеля", rng) == "Трансфер до отеля"
    for _ in range(200):
        out = drop_word("Мороженое Эскимо шоколадное без сахара", rng)
        assert out.split()[0] == "Мороженое"


def test_truncate_widths_spike_at_register_width():
    rules = load_rules()
    rng = random.Random(1)
    text = "а" * 60
    lens = [len(truncate(text, rng, rules, (24, 40))) for _ in range(3000)]
    share_22 = lens.count(22) / len(lens)
    share_35 = lens.count(35) / len(lens)
    assert 0.35 < share_22 < 0.5 and 0.15 < share_35 < 0.3


# ------------------------------------------------------------------ step 02 end to end on a tiny fake extract
def test_step02_cleans_and_remaps_an_old_extract(tmp_path, monkeypatch):
    import importlib
    import pandas as pd

    step02 = importlib.import_module("ml.scripts.02_build_category_dataset")
    monkeypatch.setattr(step02, "PROCESSED", tmp_path)
    pd.DataFrame([
        # old (wrong) label drinks, but the tags say dairy -> remapped by the CURRENT map
        {"code": "1", "name": "Кефир 3,2%", "subcategory": "food.drinks",
         "tags": "en:beverages,en:dairies,en:dairy-drinks,en:fermented-milk-products", "lang": "ru", "countries": "en:russia"},
        {"code": "2", "name": "Хліб вівсяний з льоном", "subcategory": "food.bakery", "tags": "en:breads", "lang": "", "countries": ""},
        {"code": "3", "name": "Батон &quot;Семейный&quot;", "subcategory": "food.bakery", "tags": "en:breads", "lang": "", "countries": ""},
        {"code": "4", "name": "Хлеб ржаной", "subcategory": "food.bakery", "tags": "en:breads", "lang": "", "countries": "en:bulgaria"},
        {"code": "5", "name": "Квашеная капуста", "subcategory": "food.dairy", "tags": "en:fermented-foods", "lang": "", "countries": ""},
    ]).to_csv(tmp_path / "off_food.csv", index=False)
    report = GuardReport()
    out = step02.load_off_clean(report)
    got = {(t, s) for t, s in zip(out["text_clean"], out["subcategory"])}
    assert got == {("Кефир 3,2%", "food.dairy"), ('Батон "Семейный"', "food.bakery")}
    assert report.counts["off:remap_changed_label"] >= 1
    assert report.counts["off:remap_unmapped_dropped"] == 1      # fermented-foods is no longer mapped to dairy
    assert report.counts["off:reject:ukrainian_or_other"] == 1
    assert report.counts["off:reject:country"] == 1


# ------------------------------------------------------------------ second real run (2026-10-28)
def test_country_is_a_block_list_not_an_allow_list():
    # imported goods with Russian names are filed under other countries - they must stay
    assert reject_reason("Кокосовая вода с соком личи", "en:thailand") is None
    assert reject_reason("Вода минеральная природная питьевая", "en:world") is None
    assert reject_reason("Шампоан", "en:bulgaria") == "country"
    assert reject_reason("Шампоан", "en:bulgaria,en:russia") is None


@pytest.mark.parametrize("name", [
    "Морквяно-рисовий суп з телятиною", "Суп гречаний грибний", "Молоко згущене 8.5% незбиране з цукром",
    "Печиво цукрове з ароматом пряженого молока",
])
def test_more_ukrainian_names_rejected(name):
    assert reject_reason(name) == "ukrainian_or_other"


@pytest.mark.parametrize("name", ["Мясо по-русски", "Синий кит", "Суп с грибами", "Печенье овсяное", "Рис длиннозерный"])
def test_russian_names_not_hit_by_ukrainian_markers(name):
    assert reject_reason(name) is None


def test_map_second_run_fixes():
    food = load_map()["food"]
    # milk chocolate carries en:cocoa-and-its-products and en:beverages-like ancestors: sweets, not drinks
    assert map_tags(["en:cocoa-and-its-products", "en:chocolates", "en:beverages"], food) == "food.sweets_snacks"
    assert map_tags(["en:snacks", "en:crackers"], food) == "food.sweets_snacks"          # Кириешки
    assert map_tags(["en:fruits-and-vegetables-based-foods", "en:sauces"], food) == "food.grocery"   # ketchup
    assert map_tags(["en:breads"], food) == "food.bakery"


@pytest.mark.parametrize("name,sub,expected", [
    ("Творожок суперкид земляника", "food.sweets_snacks", "food.dairy"),      # 'суперкид' is not 'суп'
    ("Шоколад молочный Strawberry cheesecake", "food.drinks", "food.sweets_snacks"),
    ("Печенье Chocoline глазированное с арахисом", "food.grocery", "food.sweets_snacks"),
    ("Кетчуп шашлычный", "food.produce", "food.grocery"),
    ("Бумага туалетная трехслойная", "clothes_beauty.cosmetics", "health.hygiene"),
    ("Торт «Птичье молоко»", "food.sweets_snacks", "food.sweets_snacks"),        # not dairy
    ("Вафли с ароматом топлёного молока", "food.sweets_snacks", "food.sweets_snacks"),
    ("Горячий шоколад", "food.drinks", "food.drinks"),
    ("Суп с творогом", "food.ready", "food.ready"),
])
def test_guard_second_run(name, sub, expected):
    assert apply_guards(name, sub)[0] == expected


def test_variants_are_independent_draws_not_forced_to_k():
    # the sampler no longer resamples until k distinct lines are found, so a name that is rarely changed
    # may return fewer than k variants - and never a copy of the clean name
    out = noisy_variants("Хлеб", 12, random.Random(0), "goods", unit_kind="none")
    assert len(out) <= 12 and "Хлеб" not in out and len(set(out)) == len(out)


def test_clip_words_keeps_the_name_identifiable():
    from ml.receipt_noise import clip_words
    rng = random.Random(0)
    assert clip_words("Экшн-камера", rng) == "Экшн-камера"          # one word: untouched
    for _ in range(300):
        out = clip_words("Стиральный порошок автомат", rng)
        assert len(out.split()) == 3 and any(w.isalpha() and not w.endswith(".") for w in out.split())


# --------------------------------------------------------------------------- third Colab run

from ml.dataset_builder import build_dataset_min_rows  # noqa: E402
from ml.name_clean import learn_foreign_lexicon  # noqa: E402
from ml.receipt_noise import add_prefix, noisy_variant_ex  # noqa: E402
from ml.seed_generator import generate_seed_clean, load_taxonomy  # noqa: E402
from ml.style_stats import compare, judged  # noqa: E402


@pytest.mark.parametrize("name", [
    'Сир плавлений "Дружба"', 'Кава мелена "Etiopia Arabica Selection"', "кленов сироп фатория", "Пуешки хапки",
    "драже ягоди", "Ядки", "Цейлонски чай", "Чипсы с вкус на сметана", "Ковбаса варена", "Згущене молоко",
    "Борошно пшеничне", "Қазақстан сүті"])
def test_third_run_foreign_names_rejected(name):
    assert reject_reason(name) is not None


@pytest.mark.parametrize("name", [
    "Солянка по-деревенски", "Картофель по-деревенски", "Мясо по-французски", "Сыр плавленый", "Масло топленое",
    "Печенье овсяное", "Яйцо куриное С1", "Творожок суперкид", "Ядро миндаля пастеризованное", "Мороженое Баскин Роббинс",
    "Чипсы со вкусом сыра", "Объём 1л", "Съёмный чехол"])
def test_third_run_russian_names_kept(name):
    assert reject_reason(name) is None


def test_language_tag_rejects_non_russian_entries():
    assert reject_reason("Молоко 3,2%", lang="bg") == "language_tag"
    assert reject_reason("Молоко 3,2%", lang="ru") is None
    assert reject_reason("Молоко 3,2%", lang="main") is None


def test_learned_foreign_lexicon_flags_only_words_unique_to_foreign_names():
    rows = []
    for i in range(8):   # "фурмеш" always sits in names that are certainly Bulgarian (letter ъ in the same name)
        rows.append((f"Пълно фурмеш {i} вид", "en:bulgaria", "bg"))
    for i in range(30):  # "молоко" is in certainly-Russian names, also in some Bulgarian ones
        rows.append((f"Молоко коровье {i}", "en:russia", "ru"))
    for i in range(6):
        rows.append((f"Мляко молоко {i}", "en:bulgaria", "bg"))
    lex = learn_foreign_lexicon(rows)
    assert "фурмеш" in lex
    assert "молоко" not in lex
    assert reject_reason("Фурмеш замороженный", lexicon=lex) == "foreign_lexicon"
    assert reject_reason("Молоко коровье", lexicon=lex) is None


def test_process_rows_accepts_lang_and_lexicon():
    rows = [("Сыр Российский", "food.dairy", "en:russia", "off", "ru"),
            ("Сыр Российский", "food.dairy", "en:russia", "off", "bg"),
            ("Хапка сыр", "food.dairy", "en:russia", "off", "ru")]
    kept = process_rows(rows, lexicon={"хапка"})
    assert kept == [("Сыр Российский", "food.dairy")]


@pytest.mark.parametrize("name,start,expected", [
    ("Шоколадное Мороженое С Шоколадным Соусом", "food.sweets_snacks", "food.sweets_snacks"),
    ("Шоколадное Мороженое С Шоколадным Соусом", "food.grocery", "food.grocery"),
    ("Студень красная цена Сытный", "food.sweets_snacks", "food.meat_fish"),
    ("Средство для стирки порошкообразное универсальное «BiMAX COLOR AUTOMAT»", "clothes_beauty.cosmetics", "home.chemicals"),
    ("цинк хелат 25мг", "clothes_beauty.cosmetics", "health.pharmacy"),
    ("Крем «A, E витамины» для лица и тела", "clothes_beauty.cosmetics", "clothes_beauty.cosmetics"),
    ("Зубная паста с витаминами", "clothes_beauty.cosmetics", "health.hygiene"),
    ("Набор вешалок деревянных", "clothes_beauty.clothes", "home.household"),
    ("Подсластитель таблетки, 1200 шт.", "food.drinks", "food.grocery"),
])
def test_third_run_guards(name, start, expected):
    assert apply_guards(name, start)[0] == expected


def test_guards_are_a_fixed_point_without_cycles():
    """Applying the guards to their own output must change nothing (that is what Stage 1 check 7 tests), and
    no pair of rules may send a name back and forth. Names are built from every pair of rule stems."""
    from ml.name_clean import _apply_once, load_guards
    g = load_guards()
    rules = g["reassign"]
    words = lambda r: list(r.get("stems", [])) + list(r.get("words", []))   # noqa: E731
    checked = 0
    for r1 in rules:
        for w1 in words(r1):
            for r2 in rules:
                for w2 in words(r2):
                    name = f"{w1} {w2}"
                    for sub in r1["from"]:
                        if sub.endswith(".*"):
                            continue
                        first, _ = apply_guards(name, sub)
                        if first is None:
                            continue
                        assert apply_guards(name, first)[0] == first, (name, sub, first)
                        cur, seen = sub, [sub]
                        for _ in range(6):
                            new, rid = _apply_once(name, cur, g)
                            if new is None or rid is None or new == cur:
                                break
                            assert new not in seen, f"rule cycle for {name!r}: {seen + [new]}"
                            seen.append(new)
                            cur = new
                        checked += 1
    assert checked > 1000


def test_add_prefix_never_glues_digits_to_the_name():
    rules = load_rules()
    rng = random.Random(0)
    for _ in range(400):
        out, digits = add_prefix("Молоко", rng, rules, p_article=1.0, p_position=0.5, p_token=0.5)
        assert digits and re.match(r"^\d+ ", out) or re.match(r"^\d+ \S", out)
        assert not re.match(r"^\d+[А-Яа-я]", out)


def test_digit_share_does_not_depend_on_how_many_variants_a_class_needs():
    """Third run: 20%..63% of the noisy lines started with a digit depending on the class, because duplicate
    removal kept mostly the high-entropy (digit) lines of short names. After the stratified selection the share
    of lines that start with a digit must be the same for a one-word name, a long name and many/few variants."""
    names = ["Тако", "Борщ", "Сыр", "Хлеб белый", "Йогурт питьевой клубничный 2,5% 290г", "Сметана 20% 400г"]
    shares = []
    for k in (3, 12):
        lines = clean_rows = 0
        for i in range(60):
            rng = random.Random(i)
            for n in names:
                out = noisy_variants(n, k, rng, "goods", unit_kind="none")
                lines += len(out) + 1             # + the clean row
                clean_rows += sum(1 for v in out if re.match(r"^\d", v))
        shares.append(clean_rows / lines)
    assert max(shares) - min(shares) < 0.06, shares
    assert all(0.10 < s < 0.30 for s in shares), shares


def test_noisy_variant_ex_flags_random_digits():
    rng = random.Random(1)
    seen = {True: 0, False: 0}
    for _ in range(300):
        text, hi = noisy_variant_ex("Сметана классическая", rng, "goods", "mass")
        seen[hi] += 1
        if hi:
            assert re.search(r"\d", text) or text != "Сметана классическая"
    assert seen[True] > 20 and seen[False] > 20


def test_min_rows_raises_variants_until_the_target_is_met():
    import pandas as pd
    tax = load_taxonomy()
    rows = generate_seed_clean(random.Random(3), 2)
    clean = pd.DataFrame(rows, columns=["text_clean", "subcategory", "profile", "group"]).assign(source="seed")
    base, k0 = build_dataset_min_rows(clean, tax, min_rows=1, default_variants=4, seed=1)
    assert k0 == 4
    bigger, k1 = build_dataset_min_rows(clean, tax, min_rows=len(base) + 500, default_variants=4,
                                        max_default_variants=9, seed=1)
    assert k1 > 4 and len(bigger) >= len(base) + 500


def test_style_compare_can_skip_features_without_judging_them():
    real = {"all_upper %": 5.0, "starts_digit %": 47.0, "bracket_token %": 0.0, "has_digit %": 59.0, "unit %": 5.0,
            "mixed_cyr_lat_in_word %": 0.0, "digit_letter_swap %": 0.0, "len_mean": 20.0, "len_p90": 30.0}
    ours = dict(real, **{"starts_digit %": 18.0, "has_digit %": 35.0})
    table = compare(real, ours, skip=("starts_digit %", "has_digit %"))
    assert set(table.loc[table["ok"] == "skip", "feature"]) == {"starts_digit %", "has_digit %"}
    assert judged(table)["ok"].all()
    assert not compare(real, ours)["ok"].all()


# ---- fourth-run fixes ----------------------------------------------------------------------------------
def test_lexicon_requires_zero_russian_names():
    from ml.name_clean import learn_foreign_lexicon
    rows = [(f"Чипс {i}", "en:bulgaria", None) for i in range(6)]
    rows += [("Чипс картофельный", "en:russia", None)]
    rows += [(f"Мляко прясно {i}", "en:bulgaria", None) for i in range(6)]
    lex = learn_foreign_lexicon(rows)
    assert "мляко" in lex
    assert "чипс" not in lex


def test_lexicon_tokens_keep_ukrainian_letters():
    from ml.name_clean import lexicon_tokens
    assert "арахіс" in lexicon_tokens("Арахіс солоний")


def test_hand_cream_with_vitamin_is_not_pharmacy():
    from ml.name_clean import apply_guards
    assert apply_guards("Нежные руки Ромашка и витамин E", "clothes_beauty.cosmetics")[0] == "clothes_beauty.cosmetics"
    assert apply_guards("цинк хелат 25мг", "clothes_beauty.cosmetics")[0] == "health.pharmacy"


def test_pate_is_meat_unless_vegetarian():
    from ml.name_clean import apply_guards
    assert apply_guards('Паштет печеночный "курица"', "food.sweets_snacks")[0] == "food.meat_fish"
    assert apply_guards("Паштет вегетарианский с баклажаном", "food.sweets_snacks")[0] == "food.sweets_snacks"


def test_no_liters_on_chemicals_or_kids_goods():
    import random
    from ml.receipt_noise import noisy_variants, unit_kind_for
    assert unit_kind_for("home.chemicals") == "none"
    assert unit_kind_for("family_kids.goods") == "none"
