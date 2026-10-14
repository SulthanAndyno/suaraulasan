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
| IndoBERT-base (`indobenchmark/indobert-base-p1`, 3 epoch) | 0.923 |

Test set identik (572 ulasan, split seed 42); semua run tercatat di MLflow.

**Temuan:** IndoBERT tidak lebih baik. Dari 16 ulasan yang salah ditebak *kedua* model, hampir semuanya
komplain yang diberi bintang 4–5 ("gojek error gini pas pilih metode pembayaran", "membuka tabungan
emas gagal terus"): model menebak benar, labelnya yang salah.

**Gold set (label manual, 300 ulasan dari test split, 25 di-skip, `data/gold.csv`):**

| | vs gold |
|---|---|
| Label bintang benar | 96.4% (F1 0.963) |
| TF-IDF + LogReg | F1 0.971 |

Noise label bintang ternyata hanya ~3.6%, dan baseline sudah lebih akurat dari label yang dipakai
melatihnya. Relabel penuh dengan LLM paling banyak menutup selisih kecil itu.

## Jalankan

```bash
uv sync
uv run python -m suaraulasan.scrape --n 1000          # tambah data ke data/reviews.db
uv run python -m suaraulasan.train                    # atau --data data/sample.csv
uv run uvicorn suaraulasan.api:app --reload
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
     -d '{"text": "aplikasi sering error, saldo nyangkut"}'
```

IndoBERT (butuh GPU, ~1 menit di RTX 3050 4 GB):

```bash
uv sync --group bert
uv run python -m suaraulasan.train --data data/sample.csv       # baseline, ikut tercatat di MLflow
uv run python -m suaraulasan.train_bert --data data/sample.csv
uv run mlflow ui                                                # bandingkan run di localhost:5000
```

Docker (latih model dulu): `docker build -t suaraulasan . && docker run -p 8000:8000 suaraulasan`

## CI/CD (fase 2)

Setiap push/PR: ruff → pytest → training di `data/sample.csv` → **gagal kalau F1 macro < 0.88**.
Push ke `main`: image dipush ke `ghcr.io/<user>/suaraulasan`.
