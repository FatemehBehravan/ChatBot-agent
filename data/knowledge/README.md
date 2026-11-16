# Knowledge Base

The assistant's knowledge base is built from the company's **public** blog and help/education content at <https://alopeyk.com/>.

## What you build
An ingestion pipeline (run inside Docker, idempotent, re-runnable) that:

1. Discovers pages: `sitemap.xml` if permitted, and links from the blog / help / education sections (see `sources.yaml`).
2. Fetches politely: respect `robots.txt`, identify your bot honestly, keep a crawl delay, cap total pages, handle failures and redirects.
3. Extracts clean main text (no navigation, footers, scripts), preserving headings.
4. Stores for each document: `source_url`, `title`, `language`, `fetched_at`, `content_hash`, text.
5. Normalizes Persian text before indexing and querying (Arabic vs Persian `ي/ی`, `ك/ک`; Persian/Arabic/Latin digits; ZWNJ; extra whitespace).
6. Chunks and indexes (keyword / BM25 / embeddings / hybrid — your choice).

Retrieval must return passages **with their `source_url`**, and the assistant must cite them. If nothing relevant is retrieved, the assistant says so instead of answering from imagination.

## What is committed
- Your ingestion code.
- At most a small sample of crawled pages. The full corpus is rebuilt by a documented command and kept on a volume / in `.gitignore`.

## Offline fixture corpus
`seed/` contains a tiny **fictional** corpus used by offline tests and some evals. Its content is intentionally simple and **does not describe the real company's policies** — never present it as real information outside tests. In fixture mode the front-matter `source_url` uses the `fixture://` scheme.

Your tests must not touch the network: they use `seed/` only.
