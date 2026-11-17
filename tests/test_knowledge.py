from pathlib import Path

from src.knowledge.crawl import crawl
from src.knowledge.index import QUESTIONS, hit_rate, index_seed, search
from src.knowledge.normalize import normalize

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "knowledge" / "seed"


def test_normalize_persian_letters_digits_and_zwnj():
    assert normalize("كيك ۱۲ ۳") == "کیک 12 3"
    assert normalize("بسته‌بندی") == "بسته بندی"
    assert normalize("  چند   فاصله  ") == "چند فاصله"


def test_fixture_retrieval_hit_rate_is_complete(tmp_path):
    db_path = tmp_path / "app.db"
    added = index_seed(SEED, db_path)
    assert added == 3
    assert index_seed(SEED, db_path) == 0
    assert hit_rate(db_path) == 1.0
    for query, source_url in QUESTIONS:
        top = search(db_path, query, limit=1)
        assert top[0].source_url == source_url
        assert top[0].text


def test_crawl_respects_robots_and_does_not_store_duplicates(tmp_path):
    db_path = tmp_path / "app.db"
    pages = {
        "https://alopeyk.com/robots.txt": (
            200,
            "text/plain",
            "User-agent: *\nDisallow: /login\nSitemap: https://alopeyk.com/sitemap.xml\n",
        ),
        "https://alopeyk.com/sitemap.xml": (
            200,
            "application/xml",
            "<urlset><url><loc>https://alopeyk.com/blog/a</loc></url>"
            "<url><loc>https://alopeyk.com/login</loc></url></urlset>",
        ),
        "https://alopeyk.com/blog/a": (
            200,
            "text/html",
            "<html><title>راهنما</title><body><nav>منو</nav><p>متن راهنمای بسته‌بندی</p>"
            "<a href='/login'>ورود</a><a href='https://example.com/x'>بیرون</a></body></html>",
        ),
    }
    fetched: list[str] = []

    def fetch(url: str, _agent: str):
        fetched.append(url)
        return pages.get(url, (404, "text/plain", ""))

    stored = crawl(ROOT / "data", db_path, fetch=fetch, sleep=lambda _seconds: None)
    assert stored == 1
    assert "https://alopeyk.com/login" not in fetched
    assert "https://example.com/x" not in fetched
    assert crawl(ROOT / "data", db_path, fetch=fetch, sleep=lambda _seconds: None) == 0
    passages = search(db_path, "راهنمای بسته بندی", limit=1)
    assert passages[0].source_url == "https://alopeyk.com/blog/a"
    assert "منو" not in passages[0].text
