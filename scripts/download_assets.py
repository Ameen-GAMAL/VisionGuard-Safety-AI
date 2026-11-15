"""Download only assets that match the published integrity manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK = PROJECT_ROOT / "configs" / "assets.lock.json"


@dataclass(frozen=True)
class Asset:
    path: str
    url: str
    size_bytes: int
    sha256: str
    group: str


def load_assets() -> list[Asset]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["schema_version"] != 1:
        raise ValueError("Unsupported asset-lock schema")
    assets = [Asset(**row) for row in lock["assets"]]
    if len({asset.path for asset in assets}) != len(assets):
        raise ValueError("Duplicate paths in the asset lock")
    for asset in assets:
        if not asset.url.startswith("https://") or not re.fullmatch(r"[0-9a-f]{64}", asset.sha256):
            raise ValueError(f"Invalid URL or SHA-256 for {asset.path}")
        if type(asset.size_bytes) is not int or asset.size_bytes < 1:
            raise ValueError(f"Invalid byte size for {asset.path}")
    return assets


def verify(path: Path, asset: Asset) -> None:
    if path.stat().st_size != asset.size_bytes:
        raise ValueError(f"{asset.path} has an unexpected byte size")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != asset.sha256:
        raise ValueError(f"{asset.path} has an unexpected SHA-256")


def ensure_asset(root: Path, asset: Asset, *, verify_only: bool = False) -> str:
    root = root.resolve()
    destination = (root / asset.path).resolve()
    if not destination.is_relative_to(root):
        raise ValueError(f"Asset path escapes the repository: {asset.path}")
    if destination.exists():
        verify(destination, asset)
        return "verified existing"
    if verify_only:
        raise FileNotFoundError(f"{asset.path} is missing")

    destination.parent.mkdir(parents=True, exist_ok=True)
    pending: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".download", delete=False
        ) as output:
            pending = Path(output.name)
            with urlopen(asset.url, timeout=60) as response:
                if not response.geturl().startswith("https://"):
                    raise ValueError(f"Non-HTTPS redirect rejected for {asset.path}")
                size = 0
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    size += len(chunk)
                    if size > asset.size_bytes:
                        raise ValueError(f"{asset.path} exceeds its expected byte size")
                    output.write(chunk)
        verify(pending, asset)
        # An exclusive same-volume link publishes the verified file without
        # overwriting an existing asset if another run finished first.
        try:
            destination.hardlink_to(pending)
        except FileExistsError:
            verify(destination, asset)
            return "verified existing"
        return "downloaded and verified"
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("sample", "models", "field", "all"), default="sample",
                        help="Sample includes the two models and the default clip")
    parser.add_argument("--verify-only", action="store_true", help="Check existing bytes without connecting")
    args = parser.parse_args()

    groups = {
        "sample": {"sample", "models"}, "models": {"models"},
        "field": {"field"}, "all": {"sample", "models", "field"},
    }[args.group]
    for asset in load_assets():
        if asset.group not in groups:
            continue
        try:
            status = ensure_asset(PROJECT_ROOT, asset, verify_only=args.verify_only)
        except (OSError, ValueError) as error:
            parser.exit(1, f"Asset check failed: {error}\nMove any mismatched file aside before retrying.\n")
        print(f"{status}: {asset.path} ({asset.size_bytes:,} bytes)")


if __name__ == "__main__":
    main()
