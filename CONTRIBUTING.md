# Contributing

Use a feature branch and describe the affected pipeline stages. Include measured evidence or a generated fixture for changed detection or safety behavior. Keep raw clips, weights, generated reports, database files, logs and credentials out of the repository. Use [private reporting](SECURITY.md) for security issues.

Install `requirements-dev.txt`, install the package in editable mode, and run `python -m pytest`. Add `requirements-dashboard.txt` to run the dashboard smoke test. The development and dashboard tests use generated fixtures and need no live footage or model weights.

Project credits are recorded in [the repository’s history](https://github.com/Ameen-GAMAL/VisionGuard-Safety-AI/commits/main/) and package metadata. Confirm each contributor’s role before attributing team work in portfolio or release material. Third-party assets have separate [terms](THIRD_PARTY_NOTICES.md).
