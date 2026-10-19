# SuaraUlasan

[![ci](https://github.com/SulthanAndyno/suaraulasan/actions/workflows/ci.yml/badge.svg)](https://github.com/SulthanAndyno/suaraulasan/actions/workflows/ci.yml)
[![retrain](https://github.com/SulthanAndyno/suaraulasan/actions/workflows/retrain.yml/badge.svg)](https://github.com/SulthanAndyno/suaraulasan/actions/workflows/retrain.yml)

Analisis sentimen ulasan Play Store aplikasi Indonesia (Gojek, BRImo, MyTelkomsel), dari scraping sampai
model yang dilatih ulang otomatis tiap minggu dengan *quality gate* berbasis label manual.

**Demo:** https://sulthanandyno.github.io/suaraulasan/ (model jalan di browser, tanpa server)

## Ringkasan

- **30 ribu ulasan asli** di-scrape dari Play Store, bertambah otomatis tiap Senin.
- **Gold set 400 ulasan berlabel manual** (300 acak + 100 kasus sulit) sebagai patokan evaluasi yang
  tidak pernah ikut dilatih.
- **Model:** TF-IDF kata + karakter dengan Logistic Regression. F1 **±0.97** di gold acak, setara
  IndoBERT tapi 200× lebih kecil (2,2 MB) dan 20× lebih cepat. Skor terbaru ada di `docs/metrics.json`.
- **Label LLM terarah** (Qwen3 4B lokal) memperbaiki ulasan bintang 5 yang isinya keluhan. F1 kasus sulit
  naik **0.690 → 0.771** tanpa menurunkan kasus biasa.
- **Retrain mingguan** via GitHub Actions; model baru ditolak bila skor gold turun.

## Arsitektur

```
              ┌─────────────── retrain.yml (tiap Senin 08.00 WIB) ───────────────┐
Play Store ──scrape.py──▶ reviews.csv ──train.py──▶ model.joblib ──gate gold──▶ export_web.py ──▶ demo (GitHub Pages)
                              ▲                         │
          relabel.py (LLM, lokal/GPU)                   └──▶ api.py ──▶ POST /predict (Docker, GHCR)
          label.py (gold set manual)
```

| File | Fungsi |
|---|---|
| `suaraulasan/scrape.py` | Ambil ulasan terbaru per aplikasi ke SQLite, ekspor ke `data/reviews.csv` (duplikat diabaikan) |
| `suaraulasan/train.py` | Label dari bintang (+ timpaan label LLM), split tetap, latih model, gate F1 test split |
| `suaraulasan/label.py` | Alat labeling gold set di terminal, evaluasi vs gold, gate gold untuk CI |
| `suaraulasan/relabel.py` | Label ulang kasus sulit dengan LLM lokal (Ollama) |
| `suaraulasan/train_bert.py` | Fine-tune IndoBERT sebagai pembanding, tercatat di MLflow |
| `suaraulasan/export_web.py` | Ekspor model ke `docs/model.json` untuk demo |
| `suaraulasan/api.py` | FastAPI `POST /predict` |
| `docs/` | Demo statis: `index.html`, `predict.js` (port sklearn ke JS, dites setara), `metrics.json` (skor gold terakhir) |
| `data/` | `reviews.csv` (data), `sample.csv` (subset kecil untuk CI), `gold.csv` (label manual), `llm_labels.csv` |

## Data

- **Sumber:** ulasan publik Play Store berbahasa Indonesia via `google-play-scraper`. Nama pengguna tidak disimpan.
- **Isi (15 Okt 2026):** ±30 ribu ulasan, ±10 ribu per aplikasi, ID unik semua, tanpa teks kosong.
- **Label lemah dari bintang:** 1–2 negatif, 4–5 positif, 3 dibuang (ambigu).
- **Catatan:**
  - 7.092 teks kembar adalah ulasan pendek dari pengguna berbeda ("mantap" 920×, "bagus" 864×), bukan duplikat scrape.
  - Rentang waktu tidak sama per aplikasi (Gojek Jun–Okt, MyTelkomsel Sep–Okt 2026) karena yang diambil adalah N ulasan terbaru.
  - Sekitar 3% berbahasa Inggris, kebanyakan pendek ("good", "nice").

## Labeling

**Aturan** (dipakai untuk gold set dan prompt LLM):

| Isi ulasan | Label |
|---|---|
| Ada keluhan atau masalah apa pun, walau ada pujian ("bagus cuma jaringan kurang lancar") | negatif |
| Bertanya atau minta tolong karena ada masalah ("kenapa gak bisa login?") | negatif |
| Puas atau memuji, termasuk saran bernada senang ("udah bagus, tambahin dark mode dong") | positif |
| Bukan opini dan tidak ada masalah ("promo bulan ini apa aja?") | skip |

**Gold set** (`data/gold.csv`), dilabel tanpa melihat bintang:
- **Acak:** 300 ulasan dari test split, 25 skip.
- **Kasus sulit:** 100 ulasan bintang 4–5 yang berisi kata keluhan, 10 skip. Labeling pertama tidak
  konsisten dengan aturan (51 skip, ulasan campuran dicap positif), jadi aturan ditegaskan lalu dilabel ulang.
- Teks gold dibuang dari data latih (termasuk duplikat seperti "baik"), jadi evaluasinya jujur.
- Gold acak dan gold sulit selalu dilaporkan terpisah, karena gold sulit sengaja tidak representatif.

## Eksperimen dan temuan

### 1. Baseline vs IndoBERT (2.858 ulasan, test split label bintang)

| Model | F1 macro |
|---|---|
| TF-IDF + LogReg | 0.927 |
| IndoBERT-base (`indobenchmark/indobert-base-p1`, 3 epoch) | 0.923 |

Dari 16 ulasan yang salah ditebak kedua model, hampir semuanya keluhan yang diberi bintang 4–5:
modelnya benar, labelnya yang salah.

### 2. Seberapa salah label bintang? (gold acak)

| | Data latih | F1 gold acak |
|---|---|---|
| Label bintang apa adanya | | 0.963 (96.4% benar) |
| TF-IDF kata | 2.858 ulasan | 0.963 |
| TF-IDF kata | 28.650 ulasan | 0.963 |
| TF-IDF kata **+ karakter** (`char_wb` 2–5) | 28.650 ulasan | **0.971** |
| IndoBERT-base, 3 epoch (7 menit di RTX 3050) | 28.650 ulasan | 0.963 |

- **Data 10× saja tidak menaikkan F1.** Yang menolong adalah n-gram karakter, yang tahan typo dan
  singkatan ("anjg", "gblk", "bgst lemot" → negatif).
- **IndoBERT vs TF-IDF:** IndoBERT sedikit unggul di test split (0.931 vs 0.923) dan jauh lebih paham kata
  kasar tanpa konteks ("goblok", "kampret lah" → negatif 98–99%), tapi seri di gold, 20× lebih lambat
  (0.59 vs 0.03 ms/ulasan, GPU), dan 476 MB vs 2,2 MB. **Demo dan API memakai TF-IDF.**

### 3. Masalahnya di definisi label, bukan model

Kedua model menilai "bagus tapi sering lemot" **positif** (IndoBERT malah 100%), karena banyak pengguna
memberi bintang 5 untuk ulasan "bagus tapi ...". Model belajar dengan benar dari label yang tidak sesuai
definisi kita. Model yang lebih besar tidak menolong.

### 4. Relabel terarah dengan LLM

Qwen3 4B lokal via Ollama, hanya untuk 1.538 ulasan bintang 4–5 yang berisi kata keluhan.

- **LLM diuji di gold dulu.** Prompt pertama F1 0.741 (hampir semua dicap negatif, termasuk "keren");
  prompt perbaikan **0.974** vs label bintang 0.963. Prompt diperbaiki setelah melihat kesalahan di gold,
  jadi angka ini sedikit optimis.
- **Hasil:** 1.304 jadi negatif, 151 tetap positif, 83 skip (tetap label bintang).

| F1 | Gold acak (275) | Gold sulit (90) |
|---|---|---|
| Label bintang apa adanya | 0.963 | 0.204 |
| Model, label bintang | 0.967 | 0.690 |
| Model, label bintang + LLM | 0.967 | **0.771** |

| Kalimat | Label bintang | + label LLM |
|---|---|---|
| bagus tapi sering lemot | positif 63% | **negatif 98%** |
| lumayan lah tapi mahal | positif 76% | **negatif 96%** |
| aplikasinya bagus, cuma kadang error | positif 90% | **negatif 93%** |
| mantap, sangat membantu | positif 100% | positif 100% |

Label LLM sama dengan label manual di 78/90 kasus sulit (label bintang: 23/90).

## Pipeline

**CI** (`ci.yml`, tiap push/PR): ruff → pytest → training di `data/sample.csv` → gagal bila F1 < 0.88.
Push ke `main` membangun image Docker ke `ghcr.io/sulthanandyno/suaraulasan`.

**Retrain** (`retrain.yml`, tiap Senin 08.00 WIB atau manual dari tab Actions):
1. Scrape ±1.000 ulasan terbaru per aplikasi, digabung ke `data/reviews.csv`.
2. Latih ulang, gate F1 test split ≥ 0.88.
3. **Gate gold:** model ditolak bila F1 gold acak **atau** gold sulit turun lebih dari setara ~3 ulasan
   (3/275 ≈ 0.011 dan 3/90 ≈ 0.033) dari `docs/metrics.json`. Toleransi awal 0.01 terlalu ketat untuk
   gold sulit (1 ulasan ≈ 0.011) dan sempat menolak model yang setara pada 19 Okt 2026.
4. Bila lolos: commit data, model demo, dan skor baru, lalu picu build GitHub Pages.

Label LLM tidak diperbarui di pipeline karena Ollama butuh GPU; `relabel.py` dijalankan lokal sesekali.

## Batasan

- Kata kasar jarang ada di Play Store (~18 dari 30 ribu ulasan, kemungkinan karena moderasi Google), jadi
  kata kasar **tanpa konteks** ("bangsat") masih bisa salah. Dalam kalimat ("aplikasi goblok") sudah benar.
- Gold set kecil: selisih 1–3 ulasan (≈0.01 di gold acak, ≈0.03 di gold sulit) belum tentu bermakna.
- Ulasan baru dari pipeline mingguan hanya memakai label bintang sampai `relabel.py` dijalankan lagi.
- Demo menampilkan "ragu" di bawah 60% dan "tidak dikenal" bila tak ada kata yang dikenali model.

## Menjalankan

```bash
uv sync
uv run python -m suaraulasan.scrape --n 10000 --export data/reviews.csv   # ~30 ribu ulasan
uv run python -m suaraulasan.train --data data/reviews.csv
uv run python -m suaraulasan.label --eval --models models/model.joblib    # skor vs gold
uv run python -m suaraulasan.export_web                                   # perbarui demo
python -m http.server -d docs 8000                                        # demo di localhost:8000
```

API:

```bash
uv run uvicorn suaraulasan.api:app --reload
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
     -d '{"text": "aplikasi sering error, saldo nyangkut"}'
```

Docker (latih model dulu): `docker build -t suaraulasan . && docker run -p 8000:8000 suaraulasan`

Labeling gold set (tombol: `n` negatif, `p` positif, `s` skip, `b` batal, `q` keluar; bisa dicicil):

```bash
uv run python -m suaraulasan.label --n 300
uv run python -m suaraulasan.label --hard --n 100 --data data/reviews.csv
```

Relabel LLM (butuh [Ollama](https://ollama.com) dengan `qwen3:4b`, ±2,5 GB):

```bash
ollama serve & ollama pull qwen3:4b
uv run python -m suaraulasan.relabel --eval    # cek akurasi LLM vs gold dulu
uv run python -m suaraulasan.relabel           # tulis data/llm_labels.csv
```

IndoBERT (butuh GPU; ±7 menit untuk 30 ribu ulasan di RTX 3050 4 GB):

```bash
uv sync --group bert
uv run python -m suaraulasan.train_bert --data data/reviews.csv
uv run mlflow ui                                # bandingkan run di localhost:5000
```

## Stack

Python 3.12+, uv, scikit-learn, FastAPI, google-play-scraper, Ollama (Qwen3 4B), PyTorch + Transformers
(IndoBERT), MLflow, pytest, ruff, Docker, GitHub Actions, GitHub Pages.
