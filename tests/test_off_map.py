from ml.off_extract import has_russian, load_map, map_tags
from ml.seed_generator import load_taxonomy


def test_first_match_wins_alcohol_over_beverages():
    rules = load_map()["food"]
    assert map_tags(["en:beverages", "en:alcoholic-beverages", "en:beers"], rules) == "food.alcohol"


def test_dairy_and_meat():
    rules = load_map()["food"]
    assert map_tags(["en:dairies", "en:milks"], rules) == "food.dairy"
    assert map_tags(["en:meats-and-their-products", "en:meats"], rules) == "food.meat_fish"


def test_baby_food_goes_to_family():
    rules = load_map()["food"]
    assert map_tags(["en:baby-foods", "en:dairies"], rules) == "family_kids.goods"


def test_prefix_match_and_unknown():
    rules = load_map()["food"]
    assert map_tags(["en:cheeses-from-france"], rules) == "food.dairy"
    assert map_tags(["en:something-new"], rules) is None


def test_defaults():
    m = load_map()
    assert map_tags(["en:whatever"], m["beauty"]) == "clothes_beauty.cosmetics"
    assert map_tags(["en:toothpastes"], m["beauty"]) == "health.hygiene"
    assert map_tags([], m["pet_food"]) == "pets.food"
    assert map_tags(["en:electronics"], m["products"]) == "home.appliances"


def test_all_targets_exist_in_taxonomy():
    tax = load_taxonomy()
    for section in load_map().values():
        for rule in section:
            assert rule["to"] in tax, rule["to"]


def test_has_russian():
    assert has_russian("Молоко")
    assert not has_russian("Milk")
    assert not has_russian("ab")
