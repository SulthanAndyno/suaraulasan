import pytest
from fastapi.testclient import TestClient

from suaraulasan import api, train

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
