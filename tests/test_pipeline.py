import json

import pytest
from fastapi.testclient import TestClient

from suaraulasan import api, label, train

SAMPLE = "data/sample.csv"


def test_label_mapping():
    assert [train.label(s) for s in range(1, 6)] == [0, 0, None, 1, 1]


def test_validate_rejects_bad_data():
    with pytest.raises(ValueError, match="sedikit"):
        train.validate([("bagus", 1)] * 10)
    with pytest.raises(ValueError, match="timpang"):
        train.validate([("bagus", 1)] * 500)


def test_train_and_api(tmp_path, monkeypatch):
    rows = train.load(SAMPLE)
    train.validate(rows)
    model, metrics = train.train(rows)
    assert 0 < metrics["f1_macro"] <= 1

    path = tmp_path / "model.joblib"
    train.joblib.dump(model, path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    api.model.cache_clear()

    c = TestClient(api.app)
    r = c.post("/predict", json={"text": "aplikasinya mantap, sangat membantu"})
    assert r.status_code == 200 and r.json()["label"] in train.LABELS
    assert c.post("/predict", json={"text": ""}).status_code == 422


def test_llm_labels_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "d.csv").write_text("content,score\nbagus tapi lemot,5\nmantap,5\n")
    assert train.load("d.csv") == [("bagus tapi lemot", 1), ("mantap", 1)]
    (tmp_path / "data/llm_labels.csv").write_text("content,label\nbagus tapi lemot,negatif\nmantap,skip\n")
    assert train.load("d.csv") == [("bagus tapi lemot", 0), ("mantap", 1)]


def test_gold_gate():
    from suaraulasan.label import passes

    old = {"acak": 0.9706, "sulit": 0.7708}
    n = {"n_acak": 275, "n_sulit": 90}
    assert passes({"acak": 0.9669, "sulit": 0.7594, **n}, old)  # retrain 19 Okt: setara, beda ~1 ulasan
    assert not passes({"acak": 0.9706, "sulit": 0.70, **n}, old)
    assert not passes({"acak": 0.94, "sulit": 0.78, **n}, old)
    assert passes({"acak": 0.9, "sulit": 0.7, **n}, {})


def test_gold_floor_stops_drift():
    from suaraulasan.label import FLOOR, passes

    n = {"n_acak": 275, "n_sulit": 90}
    last_week = {"acak": 0.965, "sulit": 0.75}  # patokan sudah turun pelan-pelan
    new = {"acak": 0.963, "sulit": 0.73, **n}
    assert passes(new, last_week)  # turun kecil dari minggu lalu: lolos
    assert not passes(new, FLOOR)  # tapi total turunnya dari model 15 Okt terlalu jauh
    assert passes({"acak": 0.9669, "sulit": 0.7594, **n}, FLOOR)  # model 19 Okt tetap lolos


def test_gold_gate_uses_floor_not_spike(tmp_path, monkeypatch):
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(json.dumps({"acak": 0.9779, "n_acak": 275, "sulit": 0.8011, "n_sulit": 90}))

    # Skor normal yang lolos FLOOR tetap diterima walau di bawah spike minggu lalu
    monkeypatch.setattr(label, "gold_scores", lambda _: {"acak": 0.9779, "n_acak": 275, "sulit": 0.7631, "n_sulit": 90})
    label.gate("unused.joblib", str(metrics_file))
    assert json.loads(metrics_file.read_text())["sulit"] == 0.7631

    # Skor di bawah FLOOR tetap ditolak tanpa menimpa metrics.json
    monkeypatch.setattr(label, "gold_scores", lambda _: {"acak": 0.9779, "n_acak": 275, "sulit": 0.72, "n_sulit": 90})
    with pytest.raises(SystemExit, match="FLOOR"):
        label.gate("unused.joblib", str(metrics_file))
    assert json.loads(metrics_file.read_text())["sulit"] == 0.7631
