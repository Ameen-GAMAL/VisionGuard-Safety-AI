"""Generated frames cover the pipeline without downloads or recorded people."""

from pathlib import Path

import pytest
from PIL import Image

from scripts.synthetic_demo import FRAME_COUNT, create_run, write_replay
from visionguard.reporting.pdf import IncidentReportBuilder
from visionguard.safety.events import EventType


@pytest.fixture()
def synthetic_run(tmp_path):
    config, pipeline, result = create_run(tmp_path / "synthetic-demo")
    return config, pipeline.store, result


def test_synthetic_frames_produce_tracked_event_and_pdf_without_screenshots(synthetic_run, tmp_path):
    config, store, result = synthetic_run
    assert result.frames_processed == FRAME_COUNT
    assert result.stats["unique_counts"]["worker"] == 1
    assert result.annotated_video.stat().st_size > 0
    assert result.heatmap_image and result.heatmap_image.is_file()
    events = store.events_for_run(result.run_id)
    assert len(events) == 1
    assert events[0].event_type is EventType.PPE_VIOLATION
    assert events[0].screenshot_path is None
    assert not config.events.screenshots_dir.exists()
    report = IncidentReportBuilder(store, config.output.reports_dir).build(result.run_id)
    assert report.read_bytes().startswith(b"%PDF-")
    replay = tmp_path / "replay.gif"
    write_replay(result.annotated_video, replay)
    with Image.open(replay) as image:
        assert image.n_frames == FRAME_COUNT // 2


def test_dashboard_opens_a_synthetic_run(synthetic_run, monkeypatch):
    pytest.importorskip("streamlit", minversion="1.58.0")
    from streamlit.testing.v1 import AppTest
    from visionguard.utils import config as config_module

    config, _, _ = synthetic_run
    monkeypatch.setattr(config_module, "load_config", lambda: config)
    root = Path(__file__).resolve().parents[1]
    page = AppTest.from_file(str(root / "src" / "visionguard" / "dashboard" / "app.py")).run(timeout=30)
    assert not page.exception
    assert any("Assistant" in tab.label for tab in page.tabs)
