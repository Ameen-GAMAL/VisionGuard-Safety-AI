"""Measure detector latency and publish JSON, raw CSV, and a timing chart."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import logging
import platform
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from visionguard.utils.config import DEFAULT_CONFIG_PATH, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SIZE = 640


def file_record(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"file_name": path.name, "size_bytes": path.stat().st_size, "sha256": digest}


def load_frames(source: Path, count: int, resize_width: int | None) -> list[tuple[int, np.ndarray]]:
    """Sample evenly from a recorded clip and preserve input frame indices."""
    capture = cv2.VideoCapture(str(source))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1:
            raise ValueError("Could not read a finite frame count from the video")
        indices = np.linspace(0, total - 1, min(total, count), dtype=int)
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Could not read sampled frame {index}")
            if resize_width and frame.shape[1] > resize_width:
                scale = resize_width / frame.shape[1]
                frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * scale)))
            frames.append((int(index), frame))
        return frames
    finally:
        capture.release()


def preprocess_onnx(frame: np.ndarray) -> np.ndarray:
    resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return blob.transpose(2, 0, 1)[None]


def timed(fn, frames, *, warmup: int, repetitions: int, synchronize=lambda: None) -> tuple[dict, list[dict]]:
    """Time each sample and synchronize accelerator work before reading the clock."""
    if not frames or warmup < 0 or repetitions < 1:
        raise ValueError("Need frames, a nonnegative warm-up, and at least one repetition")
    rows = []
    for repeat in range(1, repetitions + 1):
        for _, frame in itertools.islice(itertools.cycle(frames), warmup):
            fn(frame)
        synchronize()
        for index, frame in frames:
            start = time.perf_counter_ns()
            fn(frame)
            synchronize()
            rows.append({"repetition": repeat, "frame_index": index, "latency_ms": (time.perf_counter_ns() - start) / 1e6})
    values = np.array([row["latency_ms"] for row in rows])
    mean = float(values.mean())
    summary = {
        "n_samples": len(rows),
        "mean_ms": mean,
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "frames_per_second": 1000 / mean,
    }
    return summary, rows


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL,
                                       text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def write_report(output: Path, results: list[dict], raw_rows: list[dict], manifest: dict) -> None:
    """Write evidence without copying frames, weights, or private input paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    samples = output / "latency.csv"
    with samples.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("configuration", "repetition", "frame_index", "latency_ms"))
        writer.writeheader()
        writer.writerows(raw_rows)

    positions = np.arange(len(results))
    figure, axis = plt.subplots(figsize=(10, max(3, 0.6 * len(results) + 2)))
    axis.barh(positions + 0.16, [row["p95_ms"] for row in results], height=0.32, label="p95")
    axis.barh(positions - 0.16, [row["p50_ms"] for row in results], height=0.32, label="p50")
    axis.set_yticks(positions, [row["configuration"] for row in results])
    axis.invert_yaxis()
    axis.set_xlabel("Milliseconds per frame")
    axis.set_title("VisionGuard detector benchmark")
    axis.legend()
    figure.tight_layout()
    chart = output / "latency.png"
    figure.savefig(chart, dpi=120)
    plt.close(figure)

    report = {**manifest, "results": results, "outputs": [file_record(samples), file_record(chart)]}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--pytorch-only", action="store_true")
    args = parser.parse_args()
    if args.frames < 1 or args.repetitions < 1 or args.warmup < 0:
        parser.error("--frames and --repetitions must be positive and --warmup cannot be negative")

    started = datetime.now(timezone.utc)
    output = args.output_dir or PROJECT_ROOT / "outputs" / "benchmarks" / started.strftime("%Y%m%dT%H%M%S%fZ")
    if output.exists() and any(output.iterdir()):
        parser.error("--output-dir must be new or empty")
    config = load_config(args.config)
    if isinstance(config.video.source, int):
        parser.error("The detector benchmark requires a recorded local video file")
    source = PROJECT_ROOT / config.video.source
    weights = config.detection.model_path
    if not source.is_file() or not weights.is_file():
        parser.error("The configured clip and detector weights must exist. See scripts/download_assets.py.")
    frames = load_frames(source, args.frames, config.video.resize_width)

    import torch
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    LOGGER.setLevel(logging.ERROR)
    variants, samples, skipped = [], [], []

    def measure(name, model_path, provider, scope, fn, synchronize=lambda: None):
        try:
            summary, raw = timed(fn, frames, warmup=args.warmup, repetitions=args.repetitions,
                                 synchronize=synchronize)
        except Exception as error:
            skipped.append({"configuration": name, "reason": type(error).__name__})
            print(f"Skipped {name}: {type(error).__name__}")
            return
        variants.append({"configuration": name, "provider": provider, "timed_scope": scope,
                         "model": file_record(model_path), **summary})
        samples.extend({"configuration": name, **row} for row in raw)
        print(f"{name}: p50 {summary['p50_ms']:.1f} ms, p95 {summary['p95_ms']:.1f} ms")

    gpu = not args.cpu_only and torch.cuda.is_available()
    for device, bits in ([('cuda:0', 32), ('cuda:0', 16)] if gpu else []) + [('cpu', 32)]:
        # Ultralytics caches its predictor on each instance. A fresh instance
        # prevents a preceding variant's device or precision from leaking.
        model = YOLO(str(weights))
        measure(f"PyTorch FP{bits} ({'GPU' if device.startswith('cuda') else 'CPU'})", weights, device,
                "Ultralytics preprocessing, inference and postprocessing",
                partial(model.predict, device=device, half=bits == 16, verbose=False, imgsz=IMAGE_SIZE),
                torch.cuda.synchronize if device.startswith("cuda") else lambda: None)
        del model

    if not args.pytorch_only:
        import onnxruntime as ort
        from onnxruntime.quantization import QuantType, quantize_dynamic

        def ort_run(name: str, model_path: Path, provider: str) -> None:
            try:
                session = ort.InferenceSession(str(model_path), providers=[provider])
                if session.get_providers()[0] != provider:
                    raise RuntimeError("Requested ONNX Runtime provider was not activated")
            except Exception as error:
                skipped.append({"configuration": name, "reason": type(error).__name__})
                return
            input_name = session.get_inputs()[0].name
            measure(name, model_path, provider, "Direct-resize preprocessing and ONNX forward pass; no NMS",
                    lambda frame: session.run(None, {input_name: preprocess_onnx(frame)}))

        with tempfile.TemporaryDirectory(prefix="visionguard-onnx-", dir=weights.parent) as temp:
            export_path = Path(temp) / weights.name
            shutil.copyfile(weights, export_path)
            onnx_path = Path(YOLO(str(export_path)).export(format="onnx", imgsz=IMAGE_SIZE, simplify=True))
            if gpu and "CUDAExecutionProvider" in ort.get_available_providers():
                ort_run("ONNX Runtime FP32 (GPU)", onnx_path, "CUDAExecutionProvider")
            ort_run("ONNX Runtime FP32 (CPU)", onnx_path, "CPUExecutionProvider")
            int8_path = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
            try:
                quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QUInt8)
            except Exception as error:
                skipped.append({"configuration": "ONNX Runtime INT8 (CPU)", "reason": type(error).__name__})
            else:
                ort_run("ONNX Runtime INT8 (CPU)", int8_path, "CPUExecutionProvider")

    if not variants:
        raise RuntimeError("No backend completed the benchmark")
    packages = {}
    for name in ("torch", "torchvision", "ultralytics", "opencv-python", "numpy", "onnx", "onnxruntime", "onnxruntime-gpu"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    dirty = _git("status", "--porcelain")
    write_report(output, variants, samples, {
        "schema_version": 1,
        "scope": "one detector frame at a time; excludes decoding, tracking, and all downstream safety stages",
        "started_utc": started.isoformat(),
        "commit": _git("rev-parse", "HEAD"), "worktree_dirty": bool(dirty) if dirty is not None else None,
        "inputs": {"video": file_record(source), "weights": file_record(weights), "config": file_record(args.config),
                   "benchmark_script": file_record(Path(__file__))},
        "parameters": {"frames_requested": args.frames, "frame_indices": [index for index, _ in frames],
                       "resize_width": config.video.resize_width, "input_size": IMAGE_SIZE,
                       "warmup_per_repetition": args.warmup, "repetitions": args.repetitions,
                       "cpu_only": args.cpu_only, "pytorch_only": args.pytorch_only},
        "runtime": {"python": platform.python_version(), "os": platform.platform(), "cpu": platform.processor(),
                    "torch_threads": torch.get_num_threads(), "opencv_threads": cv2.getNumThreads(),
                    "gpu": torch.cuda.get_device_name(0) if gpu else None, "cuda": torch.version.cuda,
                    "packages": packages},
        "skipped": skipped,
    })
    print(f"Evidence written to {output}")


if __name__ == "__main__":
    main()
