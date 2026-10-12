"""Baseline TF-IDF + Logistic Regression. Label dari bintang: 1-2 negatif, 4-5 positif."""

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

LABELS = ["negatif", "positif"]


def label(score: int) -> int | None:
    # ponytail: bintang sebagai weak label; bintang 3 dibuang karena ambigu. Ganti ke label manual/LLM di fase 5.
    return None if score == 3 else int(score >= 4)


def load(path: str) -> list[tuple[str, int]]:
    if path.endswith(".csv"):
        with open(path, encoding="utf-8") as f:
            raw = [(r["content"], int(r["score"])) for r in csv.DictReader(f)]
    else:
        con = sqlite3.connect(path)
        raw = con.execute("SELECT content, score FROM reviews").fetchall()
        con.close()
    return [(text.strip(), label(s)) for text, s in raw if text and label(s) is not None]


def validate(rows: list[tuple[str, int]], min_rows: int = 200, min_share: float = 0.05) -> None:
    """Gagal cepat kalau data rusak, sebelum buang waktu training."""
    if len(rows) < min_rows:
        raise ValueError(f"data terlalu sedikit: {len(rows)} < {min_rows}")
    share = sum(y for _, y in rows) / len(rows)
    if not min_share <= share <= 1 - min_share:
        raise ValueError(f"kelas timpang parah: {share:.1%} positif")


def train(rows: list[tuple[str, int]], seed: int = 42):
    X, y = zip(*rows)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(classification_report(y_te, pred, target_names=LABELS))
    return model, {"f1_macro": round(f1_score(y_te, pred, average="macro"), 4), "n": len(rows)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/reviews.db", help=".db (SQLite) atau .csv")
    p.add_argument("--out", default="models")
    p.add_argument("--min-f1", type=float, default=0.0, help="gate CI: exit 1 kalau di bawah ini")
    a = p.parse_args()

    rows = load(a.data)
    validate(rows)
    model, metrics = train(rows)

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(metrics)

    if metrics["f1_macro"] < a.min_f1:
        sys.exit(f"GAGAL gate: f1_macro {metrics['f1_macro']} < {a.min_f1}")
