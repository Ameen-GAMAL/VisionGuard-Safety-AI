import csv
import hashlib
import io
import json

import pytest

from scripts import benchmark, download_assets


class LocalResponse(io.BytesIO):
    def geturl(self):
        return "https://example.invalid/asset"


def test_asset_published_only_after_size_and_hash_verification(tmp_path, monkeypatch):
    payload = b"verified bytes"
    asset = download_assets.Asset("models/sample.pt", "https://example.invalid/asset", len(payload),
                                  hashlib.sha256(payload).hexdigest(), "models")
    monkeypatch.setattr(download_assets, "urlopen", lambda *args, **kwargs: LocalResponse(b"replaced bytes"))
    with pytest.raises(ValueError, match="SHA-256"):
        download_assets.ensure_asset(tmp_path, asset)
    assert list((tmp_path / "models").iterdir()) == []

    monkeypatch.setattr(download_assets, "urlopen", lambda *args, **kwargs: LocalResponse(payload))
    assert download_assets.ensure_asset(tmp_path, asset) == "downloaded and verified"
    assert download_assets.ensure_asset(tmp_path, asset, verify_only=True) == "verified existing"


def test_benchmark_exports_frame_scope_and_raw_evidence(tmp_path):
    observed = []
    summary, rows = benchmark.timed(observed.append, [(5, 0), (12, 1)], warmup=3, repetitions=2)
    assert observed == [0, 1, 0, 0, 1] * 2
    assert summary["n_samples"] == 4
    assert [row["frame_index"] for row in rows] == [5, 12, 5, 12]
    result = {"configuration": "fixture", **summary}
    benchmark.write_report(tmp_path, [result], [{"configuration": "fixture", **row} for row in rows], {"fixture": True})
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["fixture"] is True
    assert manifest["results"][0]["n_samples"] == 4
    assert manifest["outputs"][0] == benchmark.file_record(tmp_path / "latency.csv")
    with (tmp_path / "latency.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 4
