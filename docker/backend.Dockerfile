# Build context: repository root (see docker-compose.yml).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

# Third-party dependencies first so this layer is cached across source edits. The `neural` extra is the
# retrieval stack (sentence-transformers, faiss-cpu, torch). On Linux torch comes from the CPU-only wheel index
# ([tool.uv.sources] in pyproject.toml), so no CUDA libraries are installed.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-install-project --extra neural

# Then the project itself (a regular, non-editable install of src/neural_debugger).
COPY src src
RUN uv sync --frozen --no-dev --no-editable --extra neural

# Runtime data for retrieval: the indexed documentation, its precomputed embeddings, the retriever configs.
COPY configs/neural.yaml configs/m4.yaml configs/
COPY data/corpus data/corpus
COPY data/embeddings data/embeddings
# Milestone 5.2: official pages merged over the frozen base corpus in memory, with their own embeddings
COPY data/corpus_supplement data/corpus_supplement
COPY data/embeddings_rag data/embeddings_rag

ENV PATH="/app/.venv/bin:$PATH" HF_HOME=/opt/hf
# Bake the pinned embedding model used to encode queries (revision read from the same config the app uses), only
# the files the encoder needs, so the container never downloads anything at runtime.
RUN python -c "import yaml; from huggingface_hub import snapshot_download; \
spec = yaml.safe_load(open('configs/neural.yaml'))['embedding_models']['gte-small']; \
snapshot_download(spec['hf_id'], revision=spec['revision'], \
allow_patterns=['*.json', '*.txt', 'model.safetensors', '1_Pooling/*'])"

# No secrets live in the image: API keys are only ever read from environment variables at run time
# (e.g. `docker run -e GEMINI_API_KEY ...`). With no ND_LLM_PROVIDER the endpoint answers 503 llm_not_configured.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ND_RETRIEVER_DEVICE=cpu ND_FEEDBACK_DIR=/app/var/feedback
RUN useradd --system --no-create-home app \
 && mkdir -p /app/var/feedback && chown -R app /app/var
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"
CMD ["uvicorn", "neural_debugger.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
