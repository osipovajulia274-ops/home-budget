import numpy as np
import pytest

pytest.importorskip("sklearn")

from app.services.linear_classifier import LinearTfidfClassifier, char_ngrams
from app.services.normalize import normalize


def _fit(tmp_path, dtype):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from ml.export_baseline import export_baseline

    texts = ["Молоко 3,2% 900мл", "Кефир 1% 500мл", "Сыр Гауда 200г", "Хлеб Бородинский 300г", "Батон нарезной",
             "Булка с маком", "Бензин АИ-95 40л", "Бензин АИ-92 30л", "Борщ 1 порц.", "Салат Цезарь 1 порц.",
             "Суп грибной 250г", "Таблетки от головы", "Пластырь бактерицидный"] * 3
    y = np.array([0, 0, 0, 1, 1, 1, 3, 3, 2, 2, 2, 5, 5] * 3)           # classes 4 and 6.. absent
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True, preprocessor=normalize,
                          lowercase=False, max_features=500)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(C=10.0, max_iter=300).fit(X, y)
    info = export_baseline(vec, clf, 8, tmp_path, coef_dtype=dtype)
    return vec, clf, info


def test_char_ngrams_match_sklearn():
    from sklearn.feature_extraction.text import TfidfVectorizer
    an = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=False).build_analyzer()
    for t in ["молоко 0%  0мл", "аб", "а", "", "салат  цезарь"]:
        assert char_ngrams(t) == an(t)


def test_probabilities_match_sklearn_float32(tmp_path):
    vec, clf, info = _fit(tmp_path, np.float32)
    m = LinearTfidfClassifier.load(tmp_path)
    probe = ["Молоко пастеризованное 1л", "борщ порц", "АИ-95 бензин", "что-то совсем незнакомое zzz", ""]
    ref = np.zeros((len(probe), 8))
    ref[:, clf.classes_] = clf.predict_proba(vec.transform(probe))
    got = m.predict_proba(probe)
    assert got.shape == (5, 8)
    assert np.allclose(got[:4], ref[:4], atol=2e-5)
    assert np.allclose(got.sum(1)[:4], 1.0, atol=1e-5)
    assert info["features"] == len(vec.vocabulary_)


def test_float16_keeps_argmax(tmp_path):
    vec, clf, info = _fit(tmp_path, np.float16)
    m = LinearTfidfClassifier.load(tmp_path)
    probe = ["Молоко 3,2% 1л", "Хлеб белый", "Борщ порц.", "Бензин 95"]
    ref = clf.predict_proba(vec.transform(probe)).argmax(1)
    assert (clf.classes_[ref] == m.predict_proba(probe).argmax(1)).all()


def test_absent_classes_get_zero(tmp_path):
    _fit(tmp_path, np.float32)
    p = LinearTfidfClassifier.load(tmp_path).predict_proba(["Молоко"])
    assert p[0, 4] == 0 and p[0, 6] == 0 and p[0, 7] == 0
