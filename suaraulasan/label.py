"""Gold set: label manual sampel dari test split (tidak pernah dipakai training).

    uv run python -m suaraulasan.label --n 300           # labeling, bisa dicicil
    uv run python -m suaraulasan.label --eval            # ukur label bintang & model vs gold

Tombol: n = negatif, p = positif, s = skip (ambigu / bukan opini), b = batal terakhir, q = keluar.
"""

import argparse
import csv
import os
import random

from sklearn.metrics import f1_score

from suaraulasan.train import LABELS, load, split

GOLD = "data/gold.csv"
FIELDS = ["idx", "content", "star_label", "gold"]


def sample(data: str, n: int, seed: int = 0) -> list[tuple[int, str, int]]:
    _, X_te, _, y_te = split(load(data))
    idx = random.Random(seed).sample(range(len(X_te)), min(n, len(X_te)))
    return [(i, X_te[i], y_te[i]) for i in idx]


def read_gold(path: str = GOLD) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_gold(rows: list[dict], path: str = GOLD) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)


def annotate(data: str, n: int) -> None:
    rows = read_gold()
    done = {int(r["idx"]) for r in rows}
    todo = [s for s in sample(data, n) if s[0] not in done]
    keys = {"n": "0", "p": "1", "s": ""}
    i = 0
    while i < len(todo):
        idx, text, star = todo[i]
        print(f"\n[{len(rows) + 1}/{n}] {text}")
        k = input("n/p/s/b/q > ").strip().lower()
        if k == "q":
            break
        if k == "b" and rows:
            rows.pop()
            write_gold(rows)
            i -= 1
            continue
        if k not in keys:
            continue
        rows.append({"idx": idx, "content": text, "star_label": star, "gold": keys[k]})
        write_gold(rows)  # ponytail: tulis ulang seluruh file tiap jawaban; aman, n kecil
        i += 1
    print(f"\n{len(rows)} tersimpan di {GOLD}")


def predict(model_path: str, texts: list[str]) -> list[int]:
    if model_path.endswith(".joblib"):
        import joblib

        return list(joblib.load(model_path).predict(texts))
    from transformers import pipeline

    clf = pipeline("text-classification", model_path, truncation=True)
    return [LABELS.index(r["label"]) for r in clf(texts, batch_size=32)]


def evaluate(models: list[str]) -> None:
    rows = [r for r in read_gold() if r["gold"] != ""]
    gold = [int(r["gold"]) for r in rows]
    texts = [r["content"] for r in rows]
    star = [int(r["star_label"]) for r in rows]
    agree = sum(s == g for s, g in zip(star, gold)) / len(gold)
    print(f"gold: {len(gold)} ulasan (skip: {len(read_gold()) - len(gold)})")
    print(f"label bintang benar: {agree:.1%}  |  F1 label bintang: {f1_score(gold, star, average='macro'):.3f}")
    for m in models:
        if os.path.exists(m):
            print(f"F1 {m}: {f1_score(gold, predict(m, texts), average='macro'):.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/sample.csv")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--models", nargs="*", default=["models/model.joblib", "models/indobert"])
    a = p.parse_args()
    evaluate(a.models) if a.eval else annotate(a.data, a.n)
