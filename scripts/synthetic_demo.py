"""Generate a shareable walkthrough from drawn shapes and scripted detections."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from visionguard import pipeline as pipeline_module
from visionguard.detection.types import BoundingBox, Detection, ObjectClass, PoseObservation
from visionguard.reporting.pdf import IncidentReportBuilder
from visionguard.utils.config import AppConfig, load_config
from visionguard.utils.video import VideoReader, VideoWriter

FRAME_COUNT = 60
FPS = 10
WIDTH = 640
HEIGHT = 360
HELMET_FRAMES = 18


def scene_detections(index: int) -> list[Detection]:
    left = 80 + index * 6
    helmet_class = ObjectClass.HELMET if index < HELMET_FRAMES else ObjectClass.NO_HELMET
    return [
        Detection(BoundingBox(left, 136, left + 70, 278), ObjectClass.WORKER, 0.99),
        Detection(BoundingBox(left + 18, 137, left + 50, 161), helmet_class, 0.99),
    ]


class ScriptedDetector:
    def __init__(self, settings):
        self.frame_index = 0

    def detect(self, _frame: np.ndarray) -> list[Detection]:
        detections = scene_detections(self.frame_index)
        self.frame_index += 1
        return detections


class NoPoseEstimator:
    def __init__(self, settings, *, device):
        pass

    def estimate(self, _frame: np.ndarray) -> list[PoseObservation]:
        return []


def _draw_scene(index: int) -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH, 3), (33, 30, 24), dtype=np.uint8)
    cv2.line(frame, (45, 280), (595, 280), (115, 86, 58), 2)
    worker, helmet = scene_detections(index)
    x1, y1 = int(worker.box.x1), int(worker.box.y1)
    x2, y2 = int(worker.box.x2), int(worker.box.y2)
    cv2.rectangle(frame, (x1 + 12, y1 + 38), (x2 - 12, y2 - 26), (190, 125, 45), -1)
    cv2.rectangle(frame, (x1 + 16, y2 - 26), (x1 + 31, y2), (155, 113, 88), -1)
    cv2.rectangle(frame, (x2 - 31, y2 - 26), (x2 - 16, y2), (155, 113, 88), -1)
    cv2.circle(frame, (x1 + 35, y1 + 29), 15, (185, 175, 142), -1)
    if helmet.object_class is ObjectClass.HELMET:
        cv2.ellipse(frame, (x1 + 35, y1 + 18), (19, 13), 0, 180, 360, (73, 217, 244), -1)
    cv2.putText(frame, "SYNTHETIC WALKTHROUGH", (365, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.59, (242, 230, 209), 1)
    cv2.putText(frame, f"{index / FPS:04.1f}s  |  Generated scene + scripted detector", (45, 332),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, (202, 194, 180), 1)
    return frame


def create_run(output: Path) -> tuple[AppConfig, pipeline_module.SafetyPipeline, pipeline_module.RunResult]:
    """Run real tracking and rules with an isolated synthetic input and database."""
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("The demo output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    source = output / "synthetic.mp4"
    with VideoWriter(source, FPS, (WIDTH, HEIGHT)) as writer:
        for index in range(FRAME_COUNT):
            writer.write(_draw_scene(index))

    base = load_config()
    config = replace(
        base,
        app=replace(base.app, log_dir=output / "logs"),
        paths=replace(base.paths, data_dir=output / "data"),
        video=replace(base.video, source=str(source), frame_skip=1, resize_width=None),
        ppe=replace(base.ppe, required_equipment=("helmet",), min_observations=2, window_seconds=0.6),
        zones=replace(base.zones, definitions_file=output / "unused-zones.json"),
        proximity=replace(base.proximity, calibration_file=output / "unused-calibration.json"),
        events=replace(base.events, database_path=output / "events.db", screenshots_dir=output / "screenshots",
                       save_screenshots=False),
        assistant=replace(base.assistant, cloud_enabled=False),
        output=replace(base.output, annotated_dir=output / "videos", heatmap_dir=output / "heatmaps",
                       reports_dir=output / "reports"),
    )
    with patch.object(pipeline_module, "Detector", ScriptedDetector), patch.object(
        pipeline_module, "PoseEstimator", NoPoseEstimator
    ):
        pipeline = pipeline_module.SafetyPipeline(config)
        result = pipeline.run()
    return config, pipeline, result


def write_replay(annotated: Path, output: Path) -> None:
    """Render the annotated frames in a compact GIF that works in a README."""
    with VideoReader(str(annotated)) as reader:
        frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).quantize(colors=96)
                  for _, frame in reader.frames(frame_skip=2)]
        duration_ms = round(2000 / reader.fps)
    if not frames:
        raise ValueError("The annotated replay has no frames")
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic-demo"))
    args = parser.parse_args()
    config, pipeline, result = create_run(args.output_dir)
    report = IncidentReportBuilder(pipeline.store, config.output.reports_dir).build(result.run_id)
    gif_path = args.output_dir / "synthetic_pipeline.gif"
    write_replay(result.annotated_video, gif_path)
    print(f"Generated replay {gif_path}, incident report {report.name}, and {result.events_total} synthetic alert(s)")


if __name__ == "__main__":
    main()
