"""Ekspor baseline TF-IDF + LogReg ke JSON untuk demo statis (docs/, GitHub Pages).

    uv run python -m suaraulasan.export_web
"""

import argparse
import json

import joblib


def export(model) -> dict:
    tfidf, clf = model.steps[0][1], model.steps[1][1]
    coef = clf.coef_[0]
    vocab = {t: [round(float(tfidf.idf_[i]), 4), round(float(coef[i]), 4)] for t, i in tfidf.vocabulary_.items()}
    return {"vocab": vocab, "b": round(float(clf.intercept_[0]), 4)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/model.joblib")
    p.add_argument("--out", default="docs/model.json")
    a = p.parse_args()
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(export(joblib.load(a.model)), f, ensure_ascii=False, separators=(",", ":"))
    print(f"tersimpan di {a.out}")
