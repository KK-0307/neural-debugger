# The whole application in ONE container: the FastAPI backend (retrieval + validated RAG) serving the real Next.js frontend
# (exported to static files) from the same origin. Build context: repository root.
#
# This is what free single-container hosts run (render.yaml). docker/backend.Dockerfile (API only) plus frontend/Dockerfile
# (Next.js server) remain the docker-compose pair for local development.

FROM node:24-slim AS site
WORKDIR /w
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# NEXT_PUBLIC_API_URL="" -> the browser calls the API on its own origin. NEXT_PUBLIC_EXPORT=1 -> static export of the real app.
ENV NEXT_PUBLIC_API_URL="" NEXT_PUBLIC_EXPORT=1 NEXT_TELEMETRY_DISABLED=1
RUN npm run build && test -f out/index.html

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-install-project --extra neural

COPY src src
RUN uv sync --frozen --no-dev --no-editable --extra neural

COPY configs/neural.yaml configs/m4.yaml configs/
COPY data/corpus data/corpus
COPY data/embeddings data/embeddings
COPY data/corpus_supplement data/corpus_supplement
COPY data/embeddings_rag data/embeddings_rag
COPY --from=site /w/out /app/site

ENV PATH="/app/.venv/bin:$PATH" HF_HOME=/opt/hf
RUN python -c "import yaml; from huggingface_hub import snapshot_download; \
spec = yaml.safe_load(open('configs/neural.yaml'))['embedding_models']['gte-small']; \
snapshot_download(spec['hf_id'], revision=spec['revision'], \
allow_patterns=['*.json', '*.txt', 'model.safetensors', '1_Pooling/*'])"

# Small-memory defaults (the free tier this targets has 512 MB): one BLAS/OpenMP thread, few allocator arenas, at most two
# answers in flight; SQLite only backs /health (the app stores no answers). No secrets here: the Gemini key is only ever read from the host's environment at run time.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ND_RETRIEVER_DEVICE=cpu ND_FEEDBACK_DIR=/app/var/feedback \
    ND_STATIC_DIR=/app/site ND_ENVIRONMENT=production ND_DATABASE_URL=sqlite:////tmp/nd.sqlite3 ND_RAG_MAX_CONCURRENT=2 ND_RAG_WARM_ON_STARTUP=true \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=2
RUN useradd --system --no-create-home app \
 && mkdir -p /app/var/feedback && chown -R app /app/var
USER app

ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s --retries=3 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8000'), timeout=2).status == 200 else 1)"
CMD ["sh", "-c", "exec uvicorn neural_debugger.backend.main:app --host 0.0.0.0 --port ${PORT}"]
