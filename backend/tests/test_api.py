import json

import pytest

from app.services import scan


def test_health_reports_indexed_sample(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["datasets"] == 1
    assert body["samples"] == 12


def test_datasets_and_samples(client):
    ds = client.get("/api/datasets").json()
    assert len(ds) == 1 and ds[0]["kind"] == "region" and ds[0]["n_samples"] == 12
    assert [c["key"] for c in ds[0]["conditions"]] == ["orig", "content_swap", "context_swap", "box_blank"]
    assert len(ds[0]["arms"]) == 5
    samples = client.get(f"/api/datasets/{ds[0]['id']}/samples").json()
    assert [s["index"] for s in samples] == list(range(12))
    assert set(samples[0]["readouts"]) == {a["key"] for a in ds[0]["arms"]}
    filtered = client.get(f"/api/datasets/{ds[0]['id']}/samples", params={"q": samples[3]["head"]}).json()
    assert samples[3]["id"] in {s["id"] for s in filtered}
    assert client.get("/api/datasets/nope/samples").status_code == 404


def test_sample_and_file(client):
    s = client.get("/api/samples/cf-00").json()
    assert s["box"] == [83, 132, 377, 82]
    r = client.get("/api/samples/cf-00/file", params={"condition": "content_swap"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and len(r.content) > 1000
    assert client.get("/api/samples/cf-00/file", params={"condition": "nope"}).status_code == 404
    assert client.get("/api/samples/nope").status_code == 404


def test_annotation_roundtrip(client):
    body = {"sample_id": "cf-01", "label": "watermark", "geometry": {"type": "bbox", "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.1}}
    r = client.post("/api/samples/cf-01/annotations", json=body)
    assert r.status_code == 201
    ann = r.json()
    assert ann["source"] == "human" and ann["geometry"]["type"] == "bbox"
    assert [a["id"] for a in client.get("/api/samples/cf-01/annotations").json()] == [ann["id"]]
    assert client.post("/api/samples/cf-01/annotations", json={**body, "id": ann["id"]}).status_code == 409
    assert client.post("/api/samples/cf-02/annotations", json=body).status_code == 422
    r = client.patch(f"/api/annotations/{ann['id']}", json={"label": "logo"})
    assert r.status_code == 200 and r.json()["label"] == "logo" and r.json()["geometry"] == ann["geometry"]
    assert client.patch("/api/annotations/nope", json={"label": "x"}).status_code == 404
    assert client.delete(f"/api/annotations/{ann['id']}").status_code == 204
    assert client.get("/api/samples/cf-01/annotations").json() == []
    assert client.delete(f"/api/annotations/{ann['id']}").status_code == 404


def test_import_skips_duplicates_and_unknown_samples(client):
    rows = [
        {"id": "imp-1", "sample_id": "cf-02", "label": "a", "geometry": {"type": "point", "x": 0.5, "y": 0.5}, "source": "human"},
        {"id": "imp-1", "sample_id": "cf-02", "label": "dup", "geometry": {"type": "point", "x": 0.5, "y": 0.5}, "source": "human"},
        {"id": "imp-2", "sample_id": "nope", "label": "b", "geometry": {"type": "point", "x": 0.5, "y": 0.5}, "source": "human"},
    ]
    r = client.post("/api/annotations/import", json=rows)
    assert r.json() == {"imported": 1, "skipped": 2}
    assert "imp-1" in {a["id"] for a in client.get("/api/annotations").json()}


def test_index_is_idempotent(client):
    from sqlmodel import Session

    from app.config import get_settings
    from app.database import get_engine
    from app.services.indexer import index

    with Session(get_engine()) as s:
        assert index(s, get_settings().data_path) == {"datasets": 1, "samples": 12}
    assert client.get("/api/health").json()["samples"] == 12


def test_scan_rejects_bad_manifests(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan.load_manifest(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"version": 1, "datasets": [
        {"id": "x", "name": "x", "kind": "region", "samples": [{"id": "x-0", "phrase": "p", "box": [0, 0, 1, 1],
                                                                 "images": {"orig": "images/missing.jpg"}, "readouts": {}, "judge": {}}]}]}))
    with pytest.raises(FileNotFoundError):
        scan.load_manifest(tmp_path)


def test_stage_copies_manifest_and_images(tmp_path):
    from app.config import get_settings

    out = scan.stage(get_settings().data_path, tmp_path / "out")
    assert (out / "manifest.json").exists()
    assert len(list((out / "images").glob("*.jpg"))) == 48


def test_render_inputs_and_parse(tmp_path):
    from app.services.describe import parse_answer, render_inputs

    marked, crop = render_inputs("../data/sample/images/00_orig.jpg", {"x": 0.166, "y": 0.395, "w": 0.754, "h": 0.245})
    assert len(marked) > 5000 and len(crop) > 5000
    assert parse_answer("Label: a watermark\nDescription: Grey text across the photo.") == ("a watermark", "Grey text across the photo.")
    assert parse_answer("**Label:** Two cats.\n**Description:** They sit.") == ("Two cats", "They sit.")
    assert parse_answer("just a sentence") == ("just a sentence", "just a sentence")


def test_describe_without_model_server(client):
    r = client.post("/api/samples/cf-00/describe", json={"geometry": {"type": "bbox", "x": 0.1, "y": 0.1, "w": 0.3, "h": 0.3}})
    assert r.status_code in (503,), r.text
    assert client.post("/api/samples/cf-00/describe", json={"geometry": {"type": "point", "x": 0.1, "y": 0.1}}).status_code == 422


def test_describe_with_fake_model_server(client, monkeypatch):
    from app.services import describe as svc

    monkeypatch.setattr(svc, "model_url", lambda settings: "http://fake:1")
    monkeypatch.setattr(svc, "describe_region", lambda path, box, url: svc.Described("a bottle", "A glass bottle on the table.", "fake", "raw"))
    r = client.post("/api/samples/cf-00/describe", json={"geometry": {"type": "bbox", "x": 0.1, "y": 0.1, "w": 0.3, "h": 0.3}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["label"] == "a bottle" and body["annotation"]["source"] == "model" and body["annotation"]["note"].startswith("A glass")
    ids = {a["id"] for a in client.get("/api/samples/cf-00/annotations").json()}
    assert body["annotation"]["id"] in ids


def test_activations_without_and_with_fake_model_server(client, monkeypatch):
    from app.services import activations as svc
    from app.services import describe as dsvc

    box = {"type": "bbox", "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.3}
    assert client.post("/api/samples/cf-00/activations", json={"geometry": box}).status_code == 503
    assert client.post("/api/samples/cf-00/activations", json={"geometry": {"type": "point", "x": 0.1, "y": 0.1}}).status_code == 422
    monkeypatch.setattr(dsvc, "model_url", lambda settings: "http://fake:1")
    monkeypatch.setattr(svc, "fetch_grid", lambda path, b, layer, url, key=None: {
        "layer": layer, "dim": 3584, "vector": [0.0] * 3584, "reduced": [0.1, -0.9], "explained": [0.3, 0.1], "basis": "fake PCA", "norm": 130.0,
        "top": [{"dim": 7, "value": -80.5}], "region_tokens": [1, 4], "rows": 2, "cols": 3, "pooling": "fake", "model": "fake"})
    r = client.post("/api/samples/cf-00/activations", json={"geometry": box, "layer": 21})
    assert r.status_code == 200, r.text
    g = r.json()
    assert g["dim"] == 3584 and g["reduced"] == [0.1, -0.9] and g["region_tokens"] == [1, 4] and g["layer"] == 21
    assert "vector" not in g  # the full vector stays in the backend


def test_predict_with_trained_model(client, monkeypatch):
    import random

    import pytest

    pytest.importorskip("torch")
    from app.services import activations as svc
    from app.services import classifier
    from app.services import describe as dsvc

    box = {"type": "bbox", "x": 0.2, "y": 0.2, "w": 0.3, "h": 0.3}
    monkeypatch.setattr(dsvc, "model_url", lambda settings: "http://fake:1")
    rnd = random.Random(0)
    monkeypatch.setattr(svc, "fetch_grid", lambda path, b, layer, url, key=None: {
        "layer": layer, "dim": 3584, "vector": [rnd.gauss(0, 1) for _ in range(3584)], "region_tokens": [3, 4, 5]})
    try:
        classifier.load()
    except classifier.ClassifierUnavailable as e:
        pytest.skip(str(e))
    r = client.post("/api/samples/cf-03/predict", json={"geometry": box})
    assert r.status_code == 200, r.text
    body = r.json()
    probs = [t["prob"] for t in body["top"]]
    assert len(probs) == 5 and probs == sorted(probs, reverse=True) and 0 < sum(probs) <= 1.0001
    assert body["n_tokens"] == 3 and body["model"]["name"] == "MLP"
    assert client.post("/api/samples/cf-03/predict", json={"geometry": {"type": "point", "x": 0.1, "y": 0.1}}).status_code == 422


def test_interactive_training_on_a_tiny_split(client, tmp_path, monkeypatch):
    import time

    import numpy as np
    import pytest

    pytest.importorskip("torch")
    from app.config import get_settings
    from app.services import classifier

    try:
        classifier.load()
    except classifier.ClassifierUnavailable as e:
        pytest.skip(str(e))
    rng = np.random.default_rng(0)
    n, d = 240, 3584
    y = rng.integers(0, 20, n)
    X = (rng.normal(0, 1, (n, d)) + y[:, None] * 0.05).astype(np.float16)
    split = np.array([0] * 160 + [1] * 40 + [2] * 40, dtype=np.int8)
    path = tmp_path / "tiny.npz"
    np.savez(path, X=X, y=y, split=split, phrase=np.array([f"phrase {i}" for i in range(n)]))
    monkeypatch.setattr(get_settings(), "train_data", path)

    r = client.post("/api/train", json={"epochs": 2, "hidden": [16], "n_train": 160, "batch_size": 32, "patience": 5})
    assert r.status_code == 200, r.text
    job = r.json()["id"]
    for _ in range(300):
        snap = client.get(f"/api/train/{job}").json()
        if snap["status"] not in ("queued", "running"):
            break
        time.sleep(0.1)
    assert snap["status"] == "done", snap
    assert [row["epoch"] for row in snap["rows"]] == [1, 2]
    assert set(snap["result"]["metrics"]) == {"train", "val", "test"} and len(snap["result"]["examples"]) == 8
    assert client.post(f"/api/train/{job}/use").json()["in_use"].endswith(f"({job})")
    assert classifier.info()["hidden"] == [16]
    assert client.post("/api/train/reset-model").json()["in_use"] is None
    assert client.post("/api/train", json={"activation": "softplus"}).status_code == 422
