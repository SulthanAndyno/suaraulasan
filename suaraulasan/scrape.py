"""Ambil ulasan Play Store terbaru dan simpan ke SQLite (duplikat diabaikan)."""

import argparse
import csv
import os
import sqlite3

from google_play_scraper import Sort, reviews

APPS = {
    "gojek": "com.gojek.app",
    "brimo": "id.co.bri.brimo",
    "mytelkomsel": "com.telkomsel.telkomselcm",
}

SCHEMA = """CREATE TABLE IF NOT EXISTS reviews (
    id TEXT PRIMARY KEY, app TEXT, content TEXT, score INTEGER, at TEXT)"""


def fetch(app_id: str, n: int) -> list[dict]:
    rows, _ = reviews(app_id, lang="id", country="id", sort=Sort.NEWEST, count=n)
    return rows


def save(db: str, app: str, rows: list[dict]) -> int:
    con = sqlite3.connect(db)
    con.execute(SCHEMA)
    before = con.total_changes
    con.executemany(
        "INSERT OR IGNORE INTO reviews VALUES (?, ?, ?, ?, ?)",
        [(r["reviewId"], app, r["content"], r["score"], r["at"].isoformat()) for r in rows],
    )
    con.commit()
    added = con.total_changes - before
    con.close()
    return added


def import_csv(db: str, path: str) -> None:
    """Isi SQLite dari CSV yang sudah di-commit (di CI belum ada .db), supaya ekspor tidak menghapus data lama."""
    con = sqlite3.connect(db)
    con.execute(SCHEMA)
    with open(path, encoding="utf-8") as f:
        rows = [(r["id"], r["app"], r["content"], int(r["score"]), r["at"]) for r in csv.DictReader(f)]
    con.executemany("INSERT OR IGNORE INTO reviews VALUES (?, ?, ?, ?, ?)", rows)
    con.commit()
    con.close()


def export_csv(db: str, path: str) -> None:
    con = sqlite3.connect(db)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "app", "content", "score", "at"])
        w.writerows(con.execute("SELECT id, app, content, score, at FROM reviews ORDER BY at"))
    con.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000, help="jumlah ulasan per aplikasi")
    p.add_argument("--db", default="data/reviews.db")
    p.add_argument("--export", help="sekalian ekspor seluruh tabel ke CSV ini")
    a = p.parse_args()
    if a.export and os.path.exists(a.export):
        import_csv(a.db, a.export)
    for app, app_id in APPS.items():
        print(f"{app}: +{save(a.db, app, fetch(app_id, a.n))} ulasan baru")
    if a.export:
        export_csv(a.db, a.export)
