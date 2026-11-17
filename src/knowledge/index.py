"""Chunk the fixture corpus and retrieve passages with SQLite FTS5 (BM25).

Keyword search is enough for this small Persian/English corpus: the questions
share exact words with the source pages, and FTS5 needs no extra model or network.
"""

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.data.db import connect_ro, connect_rw
from src.knowledge.normalize import normalize

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  content_hash TEXT PRIMARY KEY,
  source_url   TEXT NOT NULL,
  title        TEXT NOT NULL,
  language     TEXT NOT NULL,
  fetched_at   TEXT NOT NULL,
  body         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id     INTEGER PRIMARY KEY,
  content_hash TEXT NOT NULL,
  source_url   TEXT NOT NULL,
  position     INTEGER NOT NULL,
  text         TEXT NOT NULL,
  UNIQUE (content_hash, position)
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  normalized,
  source_url UNINDEXED,
  content_hash UNINDEXED,
  tokenize = 'unicode61 remove_diacritics 0'
);
"""

# Top passage must be this source. Used by the offline hit-rate check.
QUESTIONS = (
    (
        "Can I cancel my order after the courier already picked up the package?",
        "fixture://cancellation",
    ),
    ("اقلام شکستنی رو چطور بسته‌بندی کنم؟", "fixture://packaging-fragile"),
    ("قیمت سفارش به چه چیزهایی بستگی دارد؟", "fixture://price-factors"),
)


@dataclass(frozen=True)
class Passage:
    source_url: str
    title: str
    text: str
    score: float


def index_seed(seed_dir: Path, db_path: Path, *, fetched_at: str = "2026-11-14T10:00:00+03:30") -> int:
    """Index fixture markdown. Existing content hashes are left as they are."""
    connection = connect_rw(db_path)
    added = 0
    try:
        connection.executescript(_SCHEMA)
        for path in sorted(seed_dir.glob("*.md")):
            meta, body = _parse_markdown(path)
            if add_document(
                connection,
                source_url=str(meta["source_url"]),
                title=str(meta["title"]),
                language=str(meta.get("language") or "fa"),
                fetched_at=fetched_at,
                body=body,
            ):
                added += 1
        connection.commit()
        return added
    finally:
        connection.close()


def add_document(
    connection: sqlite3.Connection,
    *,
    source_url: str,
    title: str,
    language: str,
    fetched_at: str,
    body: str,
) -> bool:
    """Insert a document and its chunks. Return False when the hash is already stored."""
    content_hash = hashlib.sha256(normalize(body).encode("utf-8")).hexdigest()
    existing = connection.execute(
        "SELECT 1 FROM documents WHERE content_hash = ?",
        (content_hash,),
    ).fetchone()
    if existing is not None:
        return False
    connection.execute(
        "INSERT INTO documents (content_hash, source_url, title, language, fetched_at, body) VALUES (?,?,?,?,?,?)",
        (content_hash, source_url, title, language, fetched_at, body),
    )
    for position, text in enumerate(_chunks(body)):
        cursor = connection.execute(
            "INSERT INTO chunks (content_hash, source_url, position, text) VALUES (?,?,?,?)",
            (content_hash, source_url, position, text),
        )
        connection.execute(
            "INSERT INTO chunks_fts (rowid, normalized, source_url, content_hash) VALUES (?,?,?,?)",
            (cursor.lastrowid, normalize(text), source_url, content_hash),
        )
    return True


def search(db_path: Path, query: str, *, limit: int = 3) -> list[Passage]:
    match = _match_query(query)
    if not match:
        return []
    connection = connect_ro(db_path)
    try:
        rows = connection.execute(
            """
            SELECT c.text, c.source_url, d.title, bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c ON c.chunk_id = chunks_fts.rowid
            JOIN documents d ON d.content_hash = c.content_hash
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (match, limit),
        ).fetchall()
    finally:
        connection.close()
    return [
        Passage(source_url=row["source_url"], title=row["title"], text=row["text"], score=float(row["score"]))
        for row in rows
    ]


def hit_rate(db_path: Path) -> float:
    """Share of fixture questions whose top passage has the expected source URL."""
    if not QUESTIONS:
        return 0.0
    hits = 0
    for query, source_url in QUESTIONS:
        passages = search(db_path, query, limit=1)
        if passages and passages[0].source_url == source_url:
            hits += 1
    return hits / len(QUESTIONS)


def _chunks(body: str, size: int = 700) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 1 > size:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks or [body.strip()]


def _match_query(query: str) -> str:
    tokens: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"\w+", normalize(query), flags=re.UNICODE):
        if len(token) < 2 or token in seen:
            continue
        seen.add(token)
        tokens.append('"' + token.replace('"', "") + '"')
        if len(tokens) == 12:
            break
    return " OR ".join(tokens)


def _parse_markdown(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path.name} is missing front matter")
    _, front_matter, body = text.split("---", 2)
    meta = yaml.safe_load(front_matter)
    if not isinstance(meta, dict) or "source_url" not in meta or "title" not in meta:
        raise ValueError(f"{path.name} front matter needs source_url and title")
    return meta, body.strip()
