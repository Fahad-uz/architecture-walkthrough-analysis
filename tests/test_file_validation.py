from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from architecture_walkthrough.config import LimitSettings
from architecture_walkthrough.security.file_validation import ensure_within_directory, validate_image_file


def test_validate_image_file_accepts_png(tmp_path: Path) -> None:
    path = tmp_path / "unsafe-user-name.png"
    Image.new("RGB", (10, 10), "white").save(path)
    result = validate_image_file(path, LimitSettings(), "image/png")
    assert result.mime_type == "image/png"
    assert result.safe_filename.endswith(".png")
    assert result.safe_filename != path.name


def test_validate_image_file_rejects_unsupported_suffix(tmp_path: Path) -> None:
    path = tmp_path / "plan.txt"
    path.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        validate_image_file(path, LimitSettings())


def test_path_traversal_rejected(tmp_path: Path) -> None:
    base = tmp_path / "jobs"
    base.mkdir()
    with pytest.raises(ValueError):
        ensure_within_directory(base, tmp_path / "outside.txt")
