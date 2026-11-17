"""Normalize Persian text before indexing and before a query."""

import re

_LETTERS = str.maketrans({"ي": "ی", "ك": "ک", "ى": "ی", "ة": "ه"})
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_ZWNJ = "\u200c"
_ZWJ = "\u200d"


def normalize(text: str) -> str:
    text = text.translate(_LETTERS).translate(_DIGITS)
    text = text.replace(_ZWNJ, " ").replace(_ZWJ, "")
    return re.sub(r"\s+", " ", text).strip()
