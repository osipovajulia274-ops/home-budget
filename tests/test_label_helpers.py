import importlib

from ml.scripts_helpers import as_list

label = importlib.import_module("ml.scripts.04_label_real_test")


def test_as_list_variants():
    assert as_list(None) == []
    assert as_list(["a", "b"]) == ["a", "b"]
    assert as_list('["x", "y"]') == ["x", "y"]
    assert as_list("one\ntwo\n\nthree") == ["one", "two", "three"]


def test_is_junk():
    assert label.is_junk("")
    assert label.is_junk("12 34")
    assert label.is_junk("СКИДКА 10%")
    assert label.is_junk("ГРУППА ТОВАРОВ 5")
    assert label.is_junk("не распознано")
    assert not label.is_junk("8361 БАНАНЫ 1КГ")
