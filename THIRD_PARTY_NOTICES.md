# Third-party sources and terms

This inventory records upstream terms. The VisionGuard source license and redistribution rights for repository screenshots require the project rights holders’ decision.

| Component or asset | Source and scope |
| --- | --- |
| PPE YOLO checkpoint | [Construction-Hazard-Detection model card](https://huggingface.co/yihong1120/Construction-Hazard-Detection) lists AGPL-3.0. VisionGuard fetches a pinned revision and checks the upstream LFS digest. |
| YOLO pose checkpoint and Ultralytics | [Versioned assets](https://github.com/ultralytics/assets/releases/tag/v8.3.0) and [Ultralytics licensing terms](https://www.ultralytics.com/license). Confirm the distribution terms that apply to the planned use. |
| Re-ID and assistant embeddings | [CLIP ViT-B/32 model card](https://huggingface.co/sentence-transformers/clip-ViT-B-32) and [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Check the terms for the exact downloaded checkpoints before redistribution. The detector benchmark does not use either embedding model. |
| Supervision | [MIT license](https://github.com/roboflow/supervision/blob/develop/LICENSE.md). |
| Optional Anthropic client and service | [Python SDK terms](https://github.com/anthropics/anthropic-sdk-python) and separate [service privacy information](https://privacy.anthropic.com/). See [PRIVACY.md](PRIVACY.md) for VisionGuard’s opt-in. |
| Steelwork, person-down, digging and forklift clips | Pexels clip IDs `11798561`, `8526604`, `3967264` and `6079419` under the [Pexels license](https://www.pexels.com/license/). See [assets.lock.json](configs/assets.lock.json) for the exact download URLs and hashes. The launcher fetches clips from their publisher and does not bundle them. Review the license’s limits before associating people in clips with incidents or publishing annotated media. |
| Generated walkthrough and smoke-test frames | `scripts/synthetic_demo.py` draws a walking figure and scripts PPE detections for application checks. The README replay contains no recorded people or external video. |

Installation profiles pin direct dependency versions. Consult the corresponding upstream licenses and the CI dependency inventory when preparing a release.
