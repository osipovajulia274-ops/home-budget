import numpy as np
import pandas as pd

from ml import eval_utils as eu

LABELS = eu.load_labels()
L2I = {s: i for i, s in enumerate(LABELS)}


def test_labels_follow_taxonomy():
    assert len(LABELS) == 48 and len(eu.category_names(LABELS)) == 13
    assert LABELS[0] == "food.dairy" and "other.unknown" in LABELS


def test_class_weights_clipped_and_normalised():
    y = np.array([0] * 1000 + [1] * 100 + [2] * 400)
    w = eu.class_weights(y, 3, clip=10.0)
    assert w.max() / w.min() <= 10.0 + 1e-6
    assert w[1] > w[2] > w[0]
    assert np.isclose((w * np.bincount(y, minlength=3)).sum() / len(y), 1.0, atol=1e-5)


def test_split_metrics_levels():
    y_true = np.array([L2I["food.dairy"], L2I["food.dairy"], L2I["eating_out.cafe"], L2I["transport.taxi"]])
    y_pred = np.array([L2I["food.dairy"], L2I["food.bakery"], L2I["food.ready"], L2I["transport.public"]])
    m = eu.split_metrics(y_true, y_pred, LABELS)
    assert m["acc_sub"] == 0.25         # only the first is exact
    assert m["acc_cat"] == 0.75         # bakery is still `food`, public is `transport`; cafe -> food is wrong
    assert m["n_classes_sub"] == 3 and m["n_classes_cat"] == 3


def test_threshold_routes_low_confidence_to_unknown():
    p = np.zeros((2, len(LABELS)))
    p[0, 0] = 0.9
    p[1, 0] = 0.3; p[1, 1] = 0.2
    pred = eu.apply_threshold(p, LABELS, 0.5)
    assert pred[0] == 0 and pred[1] == L2I["other.unknown"]


def test_real_views_unique_and_without_cafe():
    real = pd.DataFrame({"text": ["Хлеб 1", "Хлеб 2", "Борщ", "Молоко"],
                         "subcategory": ["food.bakery", "food.bakery", "eating_out.cafe", "food.dairy"]})
    y = real["subcategory"].map(L2I).to_numpy()
    v = eu.real_test_views(real, y, y, LABELS)
    assert v["all_items"]["n"] == 4
    assert v["unique_names"]["n"] == 3          # "Хлеб 1" and "Хлеб 2" normalise to the same string
    assert v["without_cafe"]["n"] == 3


def test_softmax_rows_sum_to_one():
    p = eu.softmax(np.random.default_rng(0).normal(size=(5, 48)))
    assert np.allclose(p.sum(axis=1), 1.0)


def test_confusion_by_category_shape():
    y = np.array([0, 1, 2])
    cm = eu.confusion_by_category(y, y, LABELS)
    assert cm.shape == (13, 13) and cm.sum() == 3


def test_choose_production_non_inferiority():
    from ml.eval_utils import choose_production
    cnn = {"acc_cat": 0.523, "macro_f1_cat": 0.415}
    base = {"acc_cat": 0.520, "macro_f1_cat": 0.599}
    assert choose_production(cnn, base)[0] == "tfidf_logreg_baseline"      # the result of the first real run
    assert choose_production({"acc_cat": 0.80, "macro_f1_cat": 0.70}, {"acc_cat": 0.805, "macro_f1_cat": 0.705})[0] == "char_cnn"
    assert choose_production({"acc_cat": 0.78, "macro_f1_cat": 0.70}, {"acc_cat": 0.80, "macro_f1_cat": 0.70})[0] == "tfidf_logreg_baseline"


def test_dish_key_and_dictionary_keys():
    from ml.eval_utils import dish_key, seed_dictionary_keys
    assert dish_key("1 Компот из сухофруктов 0,2") == dish_key("Компот из сухофруктов {por}")
    assert "салат оливье" in seed_dictionary_keys()


def test_real_test_views_have_holdout_and_overlap():
    import numpy as np, pandas as pd
    from ml.eval_utils import real_test_views
    labels = ["food.dairy", "eating_out.cafe"]
    real = pd.DataFrame({"text": ["Молоко", "Салат оливье 1 порц.", "Суп X", "Каша Y"], "d6_split": ["train", "train", "test", "test"]})
    y = np.array([0, 1, 1, 1])
    v = real_test_views(real, y, y.copy(), labels, {"салат оливье"})
    assert v["holdout_receipts"]["n"] == 2 and v["dev_receipts"]["n"] == 2
    assert v["without_dictionary_overlap"]["n"] == 3 and v["without_dictionary_overlap"]["excluded_items"] == 1


def test_choose_production_ensemble_and_receipt_prefix(tmp_path):
    from ml.eval_utils import choose_production, load_real_test
    cnn = {"acc_cat": 0.60, "macro_f1_cat": 0.55}
    base = {"acc_cat": 0.64, "macro_f1_cat": 0.64}
    ens_good = {"acc_cat": 0.67, "macro_f1_cat": 0.66}
    ens_half = {"acc_cat": 0.67, "macro_f1_cat": 0.645}                     # F1 gain below the margin
    assert choose_production(cnn, base, ens_good)[0] == "ensemble_cnn_baseline"
    assert choose_production(cnn, base, ens_half)[0] == "tfidf_logreg_baseline"
    p = tmp_path / "r.csv"
    p.write_text("text,subcategory,receipt_id\nМолоко,food.dairy,train_1\nХлеб,food.bakery,test_2\n", encoding="utf-8")
    df = load_real_test(p, ["food.dairy", "food.bakery"])
    assert df["d6_split"].tolist() == ["train", "test"]
