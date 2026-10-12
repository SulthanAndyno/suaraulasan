FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.24 /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY suaraulasan/ suaraulasan/
# model dihasilkan `python -m suaraulasan.train` (di CI atau lokal) sebelum docker build
COPY models/model.joblib models/model.joblib
EXPOSE 8000
# Render & host lain kasih port lewat $PORT; lokal default 8000
CMD ["sh", "-c", "exec /app/.venv/bin/uvicorn suaraulasan.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
