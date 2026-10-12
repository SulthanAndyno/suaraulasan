# SuaraUlasan

Analisis sentimen ulasan Play Store aplikasi Indonesia (Gojek, BRImo, MyTelkomsel).

```
Play Store ──scrape.py──▶ SQLite ──train.py──▶ model.joblib ──api.py──▶ POST /predict
                                     │
                         CI: lint → test → train + gate F1 → Docker → GHCR
```

## Baseline (fase 1)

TF-IDF (1–2 gram) + Logistic Regression, label dari bintang (1–2 negatif, 4–5 positif, 3 dibuang).
Data: 2.858 ulasan dari `data/sample.csv`, scrape 12 Okt 2026.

| Model | F1 macro |
|---|---|
| TF-IDF + LogReg | 0.927 |

## Jalankan

```bash
uv sync
uv run python -m suaraulasan.scrape --n 1000          # tambah data ke data/reviews.db
uv run python -m suaraulasan.train                    # atau --data data/sample.csv
uv run uvicorn suaraulasan.api:app --reload
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
     -d '{"text": "aplikasi sering error, saldo nyangkut"}'
```

Docker (latih model dulu): `docker build -t suaraulasan . && docker run -p 8000:8000 suaraulasan`

## CI/CD (fase 2)

Setiap push/PR: ruff → pytest → training di `data/sample.csv` → **gagal kalau F1 macro < 0.88**.
Push ke `main`: image dipush ke `ghcr.io/<user>/suaraulasan`.
