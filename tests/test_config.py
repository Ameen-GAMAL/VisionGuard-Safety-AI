"""Tests for the configuration loader."""

from pathlib import Path

import pytest

from visionguard.utils.config import DEFAULT_CONFIG_PATH, load_config


def test_default_config_loads() -> None:
    """The shipped config.yaml must parse into a valid AppConfig."""
    config = load_config()
    assert config.app.name == "VisionGuard Safety AI"
    assert 0.0 < config.detection.confidence_threshold < 1.0
    assert config.video.frame_skip >= 1
    assert isinstance(config.paths.data_dir, Path)
    assert config.assistant.cloud_enabled is False


def test_missing_config_raises() -> None:
    """A wrong path should fail loudly, not silently use defaults."""
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")


def test_default_config_path_exists() -> None:
    """Guards against the config file being moved without updating the loader."""
    assert DEFAULT_CONFIG_PATH.exists()


def test_quoted_cloud_flag_is_rejected(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    contents = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    config.write_text(contents.replace("cloud_enabled: false", "cloud_enabled: 'false'"), encoding="utf-8")
    with pytest.raises(ValueError, match="unquoted YAML boolean"):
        load_config(config)
