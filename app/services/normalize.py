"""Single text-normalization module shared by training (ml/) and production (app/).

Never copy this logic anywhere else: the classifier's accuracy depends on the
exact same preprocessing at training and inference time.

Pipeline (guide, section 7.1):
  1. lower(), ё -> е
  2. latin look-alikes -> cyrillic, only inside words that already contain cyrillic
  3. every digit run -> "0" (prices and article numbers must not be memorised)
  4. collapse whitespace, truncate to MAX_LEN characters
  5. fixed character vocabulary, 0 = <pad>, 1 = <unk>
"""
from __future__ import annotations

import re
import string
from typing import Iterable, List

MAX_LEN = 64
PAD_ID = 0
UNK_ID = 1

# Latin letters that look like Cyrillic ones (applied after lower()).
LATIN_TO_CYR = {
    "a": "а", "b": "в", "c": "с", "e": "е", "h": "н", "k": "к",
    "m": "м", "o": "о", "p": "р", "t": "т", "x": "х", "y": "у",
}

_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")
_CYR_RE = re.compile(r"[а-я]")
_DIGITS_RE = re.compile(r"[0-9]+")
_SPACES_RE = re.compile(r"\s+")

_CYR_ALPHABET = "абвгдежзийклмнопрстуфхцчшщъыьэюя"  # no ё: removed in step 1
_PUNCT = '.,%/-()[]+*№"'

# Order is the contract with vocab.json / the ONNX model: do not reorder.
VOCAB_CHARS: List[str] = (
    list(_CYR_ALPHABET) + list(string.ascii_lowercase) + ["0", " "] + list(_PUNCT)
)
CHAR_TO_ID = {ch: i + 2 for i, ch in enumerate(VOCAB_CHARS)}
VOCAB_SIZE = len(VOCAB_CHARS) + 2


def _fix_word(match: "re.Match[str]") -> str:
    word = match.group(0)
    if _CYR_RE.search(word):
        return "".join(LATIN_TO_CYR.get(ch, ch) for ch in word)
    return word


def normalize(text: str) -> str:
    """Return the normalised form of a receipt line (without padding)."""
    if not text:
        return ""
    s = text.lower().replace("ё", "е")
    s = _WORD_RE.sub(_fix_word, s)
    s = _DIGITS_RE.sub("0", s)
    s = _SPACES_RE.sub(" ", s).strip()
    return s[:MAX_LEN].rstrip()


def encode(text: str, max_len: int = MAX_LEN) -> List[int]:
    """Normalise and map to vocabulary ids, right-padded with PAD_ID."""
    ids = [CHAR_TO_ID.get(ch, UNK_ID) for ch in normalize(text)][:max_len]
    return ids + [PAD_ID] * (max_len - len(ids))


def encode_batch(texts: Iterable[str], max_len: int = MAX_LEN):
    """Encode many strings into an int64 numpy array of shape [B, max_len]."""
    import numpy as np

    rows = [encode(t, max_len) for t in texts]
    return np.asarray(rows, dtype=np.int64).reshape(len(rows), max_len)


def vocab_dict() -> dict:
    """Content of vocab.json (written next to the ONNX model)."""
    return {
        "pad_id": PAD_ID,
        "unk_id": UNK_ID,
        "max_len": MAX_LEN,
        "chars": VOCAB_CHARS,
        "size": VOCAB_SIZE,
    }
