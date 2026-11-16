# Multi-stage image. The runtime process drops to uid 10001 in scripts/entrypoint.py
# after preparing the state volume (named volumes are created as root).
# pytest is installed so tests can run inside this image with networking disabled.

FROM python:3.12.16-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
COPY requirements.txt .
RUN pip install --prefix=/install --no-warn-script-location -r requirements.txt

FROM python:3.12.16-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    APP_PROVIDER=stub \
    APP_ENV=dev \
    DATA_DIR=/app/data \
    STATE_DIR=/app/.state \
    CONFIG_DIR=/app/config \
    LOG_LEVEL=info \
    LOCAL_BASE_URL=http://host.docker.internal:11434/v1 \
    LOCAL_API_KEY=not-needed \
    LOCAL_MODEL=qwen2.5:7b-instruct \
    REMOTE_BASE_URL=https://api.example-provider.com/v1 \
    REMOTE_API_KEY=replace-me \
    REMOTE_MODEL=replace-with-a-low-cost-model \
    EMBEDDING_PROVIDER= \
    EMBEDDING_BASE_URL= \
    EMBEDDING_MODEL=

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --home-dir /home/app --create-home --shell /usr/sbin/nologin app

WORKDIR /app

COPY --from=builder /install /usr/local
COPY src ./src
COPY config ./config
COPY prompts ./prompts
COPY scripts ./scripts
COPY tests ./tests
COPY evals ./evals
COPY requirements.txt pytest.ini ./
COPY data ./data

RUN chown -R app:app /app

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

ENTRYPOINT ["python", "/app/scripts/entrypoint.py"]
CMD ["python", "-m", "src.main"]
