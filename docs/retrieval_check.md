# Retrieval check

Retrieval is SQLite FTS5 BM25 over normalized chunks. No embedding model is used: the fixture pages and the evaluation questions share the same words, so keyword ranking is enough and stays offline.

Persian text is normalized before both indexing and querying (Arabic `ي/ك` to Persian `ی/ک`, digits to ASCII, ZWNJ to a space).

| Question | Expected top source | Result |
|---|---|---|
| Can I cancel my order after the courier already picked up the package? | `fixture://cancellation` | hit |
| اقلام شکستنی رو چطور بسته‌بندی کنم؟ | `fixture://packaging-fragile` | hit |
| قیمت سفارش به چه چیزهایی بستگی دارد؟ | `fixture://price-factors` | hit |

Top-1 hit rate on this set: **3/3 = 1.0**.
