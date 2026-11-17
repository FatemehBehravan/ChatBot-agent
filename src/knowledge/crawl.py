"""Polite crawl of the public site. Offline tests inject their own fetch function."""

import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import re

import httpx
import yaml

from src.data.db import connect_rw
from src.knowledge.index import _SCHEMA, add_document

Fetch = Callable[[str, str], tuple[int, str, str]]


def crawl(
    data_dir: Path,
    db_path: Path,
    *,
    fetch: Fetch | None = None,
    sleep: Callable[[float], None] | None = None,
    fetched_at: str = "2026-11-14T10:00:00+03:30",
) -> int:
    """Fetch allowed pages and store new documents. The same page is not stored twice."""
    settings = yaml.safe_load((data_dir / "knowledge" / "sources.yaml").read_text(encoding="utf-8"))
    user_agent = str(settings["user_agent"])
    delay = float(settings["crawl_delay_seconds"])
    max_pages = int(settings["max_pages"])
    max_depth = int(settings["max_depth"])
    base_url = str(settings["base_url"])
    domains = {str(domain) for domain in settings["allowed_domains"]}
    excluded = [str(pattern) for pattern in settings["exclude_patterns"]]
    getter = fetch or _http_get
    pause = sleep or time.sleep

    robots_url = urljoin(base_url, "robots.txt")
    pause(delay)
    _status, _kind, robots_body = getter(robots_url, user_agent)
    parser = RobotFileParser()
    parser.parse(robots_body.splitlines())

    connection = connect_rw(db_path)
    stored = 0
    seen: set[str] = set()
    try:
        connection.executescript(_SCHEMA)
        queue = _seed_urls(getter, pause, delay, user_agent, base_url, parser, robots_body)
        while queue and stored < max_pages:
            url, depth = queue.pop(0)
            if url in seen or not _allowed(url, parser, user_agent, domains, excluded):
                continue
            seen.add(url)
            pause(delay)
            status, content_type, body = getter(url, user_agent)
            if status != 200 or "html" not in content_type.lower():
                continue
            title, text, links = _extract(body, url)
            if not text:
                continue
            if add_document(
                connection,
                source_url=url,
                title=title or url,
                language=_language(text),
                fetched_at=fetched_at,
                body=text,
            ):
                stored += 1
            if depth < max_depth:
                for link in links:
                    if link not in seen:
                        queue.append((link, depth + 1))
        connection.commit()
        return stored
    finally:
        connection.close()


def _http_get(url: str, user_agent: str) -> tuple[int, str, str]:
    try:
        response = httpx.get(
            url,
            headers={"User-Agent": user_agent},
            timeout=20,
            follow_redirects=True,
        )
    except httpx.HTTPError:
        return 0, "", ""
    return response.status_code, response.headers.get("content-type", ""), response.text


def _seed_urls(getter, pause, delay, user_agent, base_url, parser, robots_body) -> list[tuple[str, int]]:
    sitemaps = [
        line.split(":", 1)[1].strip()
        for line in robots_body.splitlines()
        if line.lower().startswith("sitemap:")
    ]
    urls: list[tuple[str, int]] = []
    for sitemap in sitemaps:
        if not parser.can_fetch(user_agent, sitemap):
            continue
        pause(delay)
        status, _kind, body = getter(sitemap, user_agent)
        if status != 200:
            continue
        urls.extend((loc, 1) for loc in _locs(body))
    if urls:
        return urls
    return [(base_url, 0)]


def _locs(xml: str) -> list[str]:
    return [item.strip() for item in re.findall(r"<loc>\s*(.*?)\s*</loc>", xml)]


def _allowed(url: str, parser: RobotFileParser, user_agent: str, domains: set[str], excluded: list[str]) -> bool:
    host = urlparse(url).netloc.lower()
    if not any(host == domain or host.endswith("." + domain) for domain in domains):
        return False
    if any(pattern in url for pattern in excluded):
        return False
    return parser.can_fetch(user_agent, url)


def _extract(html: str, page_url: str) -> tuple[str, str, list[str]]:
    from html.parser import HTMLParser

    class Extractor(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.skip = 0
            self.in_title = False
            self.title: list[str] = []
            self.parts: list[str] = []
            self.links: list[str] = []

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag in {"script", "style", "nav", "footer", "noscript"}:
                self.skip += 1
            if tag == "title":
                self.in_title = True
            if tag in {"p", "h1", "h2", "h3", "li", "br"}:
                self.parts.append("\n")
            if tag == "a":
                href = dict(attrs).get("href")
                if href:
                    self.links.append(urljoin(page_url, href).split("#", 1)[0])

        def handle_endtag(self, tag: str) -> None:
            if tag in {"script", "style", "nav", "footer", "noscript"} and self.skip:
                self.skip -= 1
            if tag == "title":
                self.in_title = False

        def handle_data(self, data: str) -> None:
            if self.in_title:
                self.title.append(data)
            if self.skip == 0:
                self.parts.append(data)

    parser = Extractor()
    parser.feed(html)
    title = " ".join("".join(parser.title).split())
    text = "\n".join(line.strip() for line in "".join(parser.parts).splitlines() if line.strip())
    return title, text, parser.links


def _language(text: str) -> str:
    persian = sum(1 for char in text if "\u0600" <= char <= "\u06ff")
    return "fa" if persian * 2 >= len(text) else "en"
