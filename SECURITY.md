# Security policy

Report security issues through [GitHub private vulnerability reporting](https://github.com/Ameen-GAMAL/VisionGuard-Safety-AI/security/advisories/new) if enabled. If private reporting is unavailable, contact a project maintainer privately and request a secure channel. Share the affected commit and a short description using generated inputs. Keep real footage, employee information, credentials and sensitive logs out of public reports.

## Deployment boundary

- Keep the local Streamlit server bound to `127.0.0.1`. The prototype has no application authentication, tenancy or role-based access control. A person who can reach the dashboard may see persisted runs and evidence.
- Obtain checkpoints only from trusted publishers. `scripts/download_assets.py` checks the expected byte count and SHA-256 before publishing a file locally. Integrity checks do not establish publisher trust. PyTorch checkpoint formats can depend on deserialization behavior, so run untrusted files only in an appropriately isolated analysis environment and keep them out of ordinary VisionGuard runs.
- Keep cloud synthesis disabled for trials by default. If authorized, review [the outbound fields and storage behavior](PRIVACY.md) and allow only necessary network destinations. Avoid adding real data or secrets to CI or demo media.
- An externally accessible deployment needs authentication, TLS, isolation, resource limits and a retention plan. Use dedicated storage and review sources that can reach video decoders and model loaders. Treat model outputs and assistant replies as review aids.
