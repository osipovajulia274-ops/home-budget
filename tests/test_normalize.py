from app.services.normalize import (
    MAX_LEN, PAD_ID, UNK_ID, VOCAB_SIZE, encode, encode_batch, normalize, vocab_dict,
)


def test_lower_and_yo():
    assert normalize("ЁЖИК В ТУМАНЕ") == "ежик в тумане"


def test_digits_collapse():
    assert normalize("8361 БАНАНЫ 1КГ") == "0 бананы 0кг"
    assert normalize("МОЛОКО 3,2% 900Г") == "молоко 0,0% 0г"


def test_latin_lookalikes_only_in_cyrillic_words():
    # "Cыр" has latin C + cyrillic letters -> fixed
    assert normalize("Cыр") == "сыр"
    # a pure latin brand must stay latin
    assert normalize("COCA COLA") == "coca cola"
    # mixed token with digits
    assert normalize("МOЛОКО") == "молоко"


def test_whitespace_and_truncate():
    assert normalize("  хлеб   белый  ") == "хлеб белый"
    long = "а" * 200
    assert len(normalize(long)) == MAX_LEN


def test_empty():
    assert normalize("") == ""
    assert normalize(None) == ""  # type: ignore[arg-type]


def test_encode_shape_and_pad():
    ids = encode("хлеб")
    assert len(ids) == MAX_LEN
    assert ids[4:] == [PAD_ID] * (MAX_LEN - 4)
    assert PAD_ID not in ids[:4]


def test_unknown_char():
    ids = encode("хлеб~")
    assert ids[4] == UNK_ID


def test_batch():
    arr = encode_batch(["молоко", "хлеб"])
    assert arr.shape == (2, MAX_LEN)
    assert arr.dtype.name == "int64"


def test_vocab_consistent():
    v = vocab_dict()
    assert v["size"] == VOCAB_SIZE == len(v["chars"]) + 2
    assert len(set(v["chars"])) == len(v["chars"])
