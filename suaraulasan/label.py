"""Gold set: label manual sampel dari test split (tidak pernah dipakai training).

    uv run python -m suaraulasan.label --n 300           # labeling, bisa dicicil
    uv run python -m suaraulasan.label --hard --n 100 --data data/reviews.csv   # kasus sulit saja
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


def hard_sample(data: str, n: int, seed: int = 1) -> list[tuple[str, str, int]]:
    """Kasus sulit: bintang 4-5 tapi ada kata keluhan (kandidat relabel LLM)."""
    from suaraulasan.relabel import candidates  # import lokal: relabel mengimpor modul ini

    pool = candidates(data)
    return [("", t, 1) for t in random.Random(seed).sample(pool, min(n, len(pool)))]


def annotate(data: str, n: int, hard: bool = False) -> None:
    rows = read_gold()
    done = {r["content"] for r in rows}
    todo = [s for s in (hard_sample(data, n) if hard else sample(data, n)) if s[1] not in done]
    keys = {"n": "0", "p": "1", "s": ""}
    i = 0
    while i < len(todo):
        idx, text, star = todo[i]
        print(f"\n[{i + 1}/{len(todo)}] {text}")
        k = input("n/p/s/b/q > ").strip().lower()
        if k == "q":
            break
        if k == "b" and i > 0:  # batal hanya untuk jawaban sesi ini
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
    from suaraulasan.relabel import COMPLAINT

    hard = [i for i, (t, s) in enumerate(zip(texts, star)) if s == 1 and COMPLAINT.search(t.lower())]
    acc = lambda pred: f"{sum(pred[i] == gold[i] for i in hard)}/{len(hard)}"
    print(f"kasus sulit (bintang 4-5 + kata keluhan): label bintang benar {acc(star)}")
    for m in models:
        if os.path.exists(m):
            pred = predict(m, texts)
            print(f"F1 {m}: {f1_score(gold, pred, average='macro'):.3f} | kasus sulit benar {acc(pred)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/sample.csv")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--hard", action="store_true", help="hanya kasus sulit; pakai --data data/reviews.csv")
    p.add_argument("--models", nargs="*", default=["models/model.joblib", "models/indobert"])
    a = p.parse_args()
    evaluate(a.models) if a.eval else annotate(a.data, a.n, a.hard)
