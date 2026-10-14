"""Label ulang ulasan bintang 4-5 yang berisi keluhan dengan LLM lokal (Ollama), sesuai aturan gold set.

    uv run python -m suaraulasan.relabel --eval        # cek akurasi LLM vs gold dulu
    uv run python -m suaraulasan.relabel               # tulis data/llm_labels.csv (bisa dicicil)

Bintang 4-5 + "bagus tapi sering lemot" = positif menurut bintang, negatif menurut aturan kita.
train.load() memakai data/llm_labels.csv untuk menimpa label bintang bila file itu ada.
"""

import argparse
import csv
import json
import os
import re
import urllib.request

from sklearn.metrics import f1_score

from suaraulasan.label import read_gold

OUT = "data/llm_labels.csv"
COMPLAINT = re.compile(
    r"\b(tapi|tp|tpi|namun|sayang|sayangnya|cuma|cuman|kurang|lemot|lambat|lelet|error|eror|gagal|bug|"
    r"mahal|tolong|kecewa|susah|ribet|lag|ngelag|nge-lag|ga bisa|gak bisa|gk bisa|tidak bisa|gabisa)\b"
)
PROMPT = """Tentukan sentimen ulasan aplikasi berikut.
- positif: pengguna puas, memuji, atau berterima kasih (contoh: "mantap", "sangat membantu").
- negatif: ada keluhan atau masalah, termasuk pujian yang diikuti keluhan (contoh: "bagus tapi sering lemot").
- skip: bukan opini (contoh: pertanyaan "kapan ada diskon").

Ulasan: {text}"""
SCHEMA = {"type": "object", "properties": {"label": {"enum": ["positif", "negatif", "skip"]}}, "required": ["label"]}


def ask(text: str, model: str, host: str) -> str:
    body = {"model": model, "prompt": PROMPT.format(text=text), "format": SCHEMA, "stream": False,
            "think": False, "options": {"temperature": 0}}
    req = urllib.request.Request(f"{host}/api/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(json.loads(r.read())["response"])["label"]


def candidates(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        texts = {r["content"].strip() for r in csv.DictReader(f) if int(r["score"]) >= 4}
    return sorted(t for t in texts if COMPLAINT.search(t.lower()))


def evaluate(model: str, host: str) -> None:
    rows = [r for r in read_gold() if r["gold"] != ""]
    pred = [ask(r["content"], model, host) for r in rows]
    gold = [int(r["gold"]) for r in rows]
    # ponytail: "skip" dari LLM dihitung sebagai label bintang (perilaku yang sama saat dipakai training)
    llm = [{"negatif": 0, "positif": 1}.get(p, int(r["star_label"])) for p, r in zip(pred, rows)]
    print(f"LLM vs gold: F1 {f1_score(gold, llm, average='macro'):.3f} | skip {pred.count('skip')}")
    hard = [i for i, r in enumerate(rows) if r["star_label"] == "1" and COMPLAINT.search(r["content"].lower())]
    if hard:
        right = sum(llm[i] == gold[i] for i in hard)
        print(f"kasus bintang positif + kata keluhan di gold: LLM benar {right}/{len(hard)}, "
              f"bintang benar {sum(gold[i] == 1 for i in hard)}/{len(hard)}")


def relabel(data: str, model: str, host: str) -> None:
    done = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            done = {r["content"]: r["label"] for r in csv.DictReader(f)}
    todo = [t for t in candidates(data) if t not in done]
    print(f"kandidat: {len(done) + len(todo)} (sudah: {len(done)})")
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["content", "label"])
        for i, t in enumerate(todo, 1):
            try:
                w.writerow([t, ask(t, model, host)])
            except (OSError, ValueError, KeyError) as e:  # satu ulasan gagal (mis. Ollama 500) tidak boleh menghentikan semuanya
                print(f"gagal, dicoba lagi di run berikutnya: {t[:60]!r} ({e})")
                continue
            f.flush()  # aman dihentikan kapan saja
            if i % 100 == 0:
                print(f"{i}/{len(todo)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/reviews.csv")
    p.add_argument("--model", default="qwen3:4b")
    p.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    p.add_argument("--eval", action="store_true")
    a = p.parse_args()
    evaluate(a.model, a.host) if a.eval else relabel(a.data, a.model, a.host)
