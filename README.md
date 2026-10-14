# SuaraUlasan

Analisis sentimen ulasan Play Store aplikasi Indonesia (Gojek, BRImo, MyTelkomsel).

**Demo:** https://sulthanandyno.github.io/suaraulasan/ (model jalan di browser, tanpa server)

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

| | Data latih | vs gold (F1) |
|---|---|---|
| Label bintang apa adanya | | 0.963 (96.4% benar) |
| TF-IDF kata + LogReg | 2.858 ulasan | 0.963 |
| TF-IDF kata + LogReg | 28.650 ulasan | 0.963 |
| TF-IDF kata **+ karakter** + LogReg | 28.650 ulasan | **0.971** |

Ulasan gold dibuang dari data latih (termasuk duplikat teks pendek seperti "baik"), jadi angka di
atas jujur. Noise label bintang hanya ~3.6%; relabel penuh dengan LLM paling banyak menutup selisih kecil itu.

**Data 10× saja tidak menaikkan F1.** Yang menolong adalah n-gram karakter (`char_wb` 2–5): tahan typo
dan singkatan ("anjg", "gblk", "bgst lemot" → negatif).

**Batasan:**
- Kata kasar jarang muncul di Play Store (~18 dari 30 ribu ulasan, kemungkinan karena moderasi Google),
  jadi kata kasar *tanpa konteks* ("bangsat") masih bisa salah. Dalam kalimat ("aplikasi goblok") sudah benar.
- TF-IDF tidak paham "tapi": "bagus tapi sering lemot" masih condong positif.
- Demo menampilkan "ragu" di bawah 60% dan "tidak dikenal" bila tak ada kata yang dikenali model.

## Jalankan

```bash
uv sync
uv run python -m suaraulasan.scrape --n 10000 --export data/reviews.csv   # ~30 ribu ulasan
uv run python -m suaraulasan.train --data data/reviews.csv   # sample.csv: subset kecil untuk CI
uv run python -m suaraulasan.export_web                       # perbarui docs/model.json untuk demo
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
